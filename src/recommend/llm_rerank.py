from __future__ import annotations
import re

from dotenv import load_dotenv
load_dotenv()

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LLMReranker:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path(".llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    def _cache_key(self, user_id: str, candidate_ids: list[str]) -> str:
        key_str = f"{user_id}:{','.join(candidate_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _build_prompt(self, user_history: list[dict], candidates: list[dict]) -> str:
        history_parts = []
        for item in user_history:
            title = item.get("title", "")
            tags = item.get("tags", "")
            history_parts.append(f"- {title} (tags: {tags})")
        history_str = "\n".join(history_parts) if history_parts else "None"

        cand_parts = []
        for c in candidates:
            cand_parts.append(f"- {c['id']}: {c.get('title','')} | {c.get('description','')}")
        cand_str = "\n".join(cand_parts)

        prompt = f"""User interaction history (completed and rated>=4):
{history_str}

Candidate courses to rerank (top 20 from hybrid):
{cand_str}

Re-rank these candidates by relevance to the user's interests and history. Return a JSON list of course_ids in order of relevance (most relevant first). Only return course_ids from the candidate list. Do not hallucinate new course_ids."""
        return prompt

    def _mock_llm_call(self, prompt: str) -> str:
        raise NotImplementedError("No API call in stub mode")

    def rerank(self, user_id, candidates, user_history=None):
        if user_history is None:
            user_history = []
        candidate_ids = [c["id"] for c in candidates]

        if not user_history:
            texts = [str(c.get("description", "")) for c in candidates]
            if not texts:
                return candidate_ids
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(texts)
                generic = tfidf_matrix.mean(axis=0)
                sims = cosine_similarity(generic, tfidf_matrix).flatten()
                ranked_idx = sims.argsort()[::-1]
                return [candidate_ids[i] for i in ranked_idx]
            except Exception:
                return candidate_ids

        cache_key = self._cache_key(user_id, candidate_ids)
        cache_path = self._cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and set(cached) == set(candidate_ids) and len(cached) == len(candidate_ids):
                    return cached
            except Exception:
                pass

        if not self.api_key:
            return candidate_ids

        prompt = self._build_prompt(user_history, candidates)

        try:
            import openai as _oai
            client = _oai.OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            response = completion.choices[0].message.content
            clean = response.strip()
            while clean and clean[0] == chr(96):
                clean = clean[1:]
            if clean.lower().startswith("json"):
                clean = clean[4:]
            while clean and clean[-1] == chr(96):
                clean = clean[:-1]
            clean = clean.strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list) and set(parsed) == set(candidate_ids) and len(parsed) == len(candidate_ids):
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f)
                return parsed
        except Exception as e:
            pass

        return candidate_ids

