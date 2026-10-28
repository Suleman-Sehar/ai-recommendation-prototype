from __future__ import annotations

import json
import os
from pathlib import Path

from src.recommend.llm_rerank import LLMReranker


class TestLLMRerank:
    def test_mock_llm_call_and_valid_parsing(self, tmp_path):
        reranker = LLMReranker(cache_dir=tmp_path)
        reranker.api_key = "test-key"
        candidates = [
            {"id": "c1", "title": "A", "description": "test"},
            {"id": "c2", "title": "B", "description": "test"},
            {"id": "c3", "title": "C", "description": "test"},
        ]
        history = [{"title": "X", "tags": "tag1"}]
        reranked = ["c3", "c1", "c2"]
        
        def mock_call(self, prompt):
            return json.dumps(reranked)
        
        reranker._mock_llm_call = mock_call.__get__(reranker, LLMReranker)
        result = reranker.rerank("user1", candidates, history)
        assert result == reranked

    def test_hallucination_detection_triggers_fallback(self, tmp_path):
        reranker = LLMReranker(cache_dir=tmp_path)
        reranker.api_key = "test-key"
        candidates = [
            {"id": "c1", "title": "A", "description": "test"},
            {"id": "c2", "title": "B", "description": "test"},
        ]
        history = [{"title": "X", "tags": "tag1"}]
        
        def mock_call(self, prompt):
            return json.dumps(["c1", "c2", "c3"])  # hallucination - extra id
        
        reranker._mock_llm_call = mock_call.__get__(reranker, LLMReranker)
        result = reranker.rerank("user1", candidates, history)
        assert result == ["c1", "c2"]  # fallback to hybrid order

    def test_caching_uses_cache_not_api(self, tmp_path):
        reranker = LLMReranker(cache_dir=tmp_path)
        reranker.api_key = "test-key"
        candidates = [
            {"id": "c1", "title": "A", "description": "test"},
            {"id": "c2", "title": "B", "description": "test"},
        ]
        history = [{"title": "X", "tags": "tag1"}]
        call_count = [0]
        
        def mock_call(self, prompt):
            call_count[0] += 1
            return json.dumps(["c2", "c1"])
        
        reranker._mock_llm_call = mock_call.__get__(reranker, LLMReranker)
        result1 = reranker.rerank("user1", candidates, history)
        result2 = reranker.rerank("user1", candidates, history)
        assert result1 == ["c2", "c1"]
        assert result2 == ["c2", "c1"]
        assert call_count[0] == 1  # only called once due to cache

    def test_cold_start_returns_without_api_call(self, tmp_path):
        reranker = LLMReranker(cache_dir=tmp_path)
        reranker.api_key = "test-key"
        candidates = [
            {"id": "c1", "title": "A", "description": "machine learning basics"},
            {"id": "c2", "title": "B", "description": "cooking recipes"},
        ]
        history = []
        call_count = [0]
        
        def mock_call(self, prompt):
            call_count[0] += 1
            return json.dumps(["c1", "c2"])
        
        reranker._mock_llm_call = mock_call.__get__(reranker, LLMReranker)
        result = reranker.rerank("user1", candidates, history)
        assert isinstance(result, list)
        assert len(result) == 2
        assert call_count[0] == 0  # no API call for cold start

    def test_stub_path_returns_hybrid_order(self, tmp_path):
        reranker = LLMReranker(cache_dir=tmp_path)
        reranker.api_key = None  # no API key
        candidates = [
            {"id": "c1", "title": "A", "description": "test"},
            {"id": "c2", "title": "B", "description": "test"},
            {"id": "c3", "title": "C", "description": "test"},
        ]
        history = [{"title": "X", "tags": "tag1"}]
        result = reranker.rerank("user1", candidates, history)
        assert result == ["c1", "c2", "c3"]  # hybrid order unchanged
