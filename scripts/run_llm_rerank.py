#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.config import CATALOG_PATH, INTERACTIONS_PATH, SPLITS_PATH, EVAL_KS
from src.data.loader import load_catalog, load_interactions, load_splits
from src.evaluate.metrics import evaluate_recommender
from src.recommend.hybrid import Hybrid
from src.recommend.llm_rerank import LLMReranker


class RerankedWrapper:
    def __init__(self, rec_fn):
        self.rec_fn = rec_fn

    def recommend(self, user_id, n=10):
        return self.rec_fn(user_id, n)


def main():
    parser = argparse.ArgumentParser(description="Run LLM reranking evaluation")
    parser.add_argument("--limit", type=int, default=85)
    parser.add_argument("--output-dir", type=str, default="reports/eval_20261026")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    catalog = load_catalog(CATALOG_PATH)
    interactions = load_interactions(INTERACTIONS_PATH)
    splits = load_splits(SPLITS_PATH)
    df = pd.DataFrame(interactions)
    df["id"] = df["id"].astype(str)
    train_ids = set(splits["train_interactions"])
    test_df = df[df["id"].isin(splits["test_holdout"])]
    train_df = df[df["id"].isin(train_ids)]
    test_users = sorted(set(test_df["user_id"].tolist()))[: args.limit]

    h = Hybrid()
    h.fit(catalog, train_df)
    reranker = LLMReranker(cache_dir=Path(".llm_cache"))
    reranker.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if args.dry_run:
        reranker.api_key = None

    cat_map = {c["id"]: c for c in catalog}
    total_cost = 0.0
    cost_per_call = 0.00015

    def rerank_recommender(user_id, n=10):
        nonlocal total_cost
        if total_cost >= 0.10:
            raise RuntimeError("Cost cap exceeded")
        hyb_top20 = h.recommend(user_id, 20, alpha=args.alpha)
        candidates = [{"id": cid, "title": cat_map[cid].get("title", ""), "description": cat_map[cid].get("description", "")} for cid in hyb_top20]
        hist = train_df[train_df["user_id"] == str(user_id)]
        hist_list = []
        for _, r in hist.iterrows():
            if r["action"] == "completed" or (r["action"] == "rated" and float(str(r.get("rating") or 0)) >= 4):
                c = cat_map.get(r["course_id"], {})
                hist_list.append({"title": c.get("title", ""), "tags": c.get("tags", "")})
        reranked = reranker.rerank(user_id, candidates, hist_list)
        if not args.dry_run and reranker.api_key:
            cache_key = reranker._cache_key(user_id, [c["id"] for c in candidates])
            cache_path = reranker._cache_path(cache_key)
            if not cache_path.exists():
                total_cost += cost_per_call
        if total_cost >= 0.10:
            raise RuntimeError("Cost cap exceeded")
        return reranked[:n]

    gt = {}
    ct = test_df[test_df["action"] == "completed"]
    rt = test_df[test_df["action"] == "rated"]
    hrt = rt[rt["rating"].astype(float) >= 4.0]
    pt = pd.concat([ct, hrt])
    for uid, group in pt.groupby("user_id"):
        gt[uid] = set(group["course_id"].tolist())

    for i, uid in enumerate(test_users):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(test_users)} users, cost so far ")
        try:
            _ = rerank_recommender(uid, 10)
        except RuntimeError as e:
            print(str(e))
            break

    results = evaluate_recommender(RerankedWrapper(rerank_recommender), test_users, gt, ks=EVAL_KS)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# LLM Rerank Evaluation", "", f"Date: {__import__('datetime').datetime.now().strftime('%Y%m%d')}", ""]
    lines.append("| Recommender | K | Precision@K | Recall@K | NDCG@K | Hit Rate@K | Users |")
    lines.append("|---|---|---|---|---|---|---|")
    for k_key, m in results.items():
        k = k_key.replace("k", "")
        lines.append(f"| llm_rerank | {k} | {m['precision']} | {m['recall']} | {m['ndcg']} | {m['hit_rate']} | {m['users_evaluated']} |")
    lines.append("")
    lines.append(f"Total cost: ")
    (out_dir / "llm_rerank.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "llm_rerank.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"Total cost: ")


if __name__ == "__main__":
    main()
