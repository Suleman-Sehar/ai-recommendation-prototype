# Decision Log — AI Recommendation Prototype

## 2026-10-26 — Phase 0
- Folder tree: src/{data,recommend,evaluate}/ layout with tests/ mirroring modules
- Stack: Python 3.11, pandas, scikit-learn, numpy, sentence-transformers, pytest
- LLM: OpenRouter via OPENAI_API_KEY + --base-url, cold-start embeddings + re-rank only, approval-gated
- Metrics: Precision@10, Recall@10, NDCG@10 vs random/popularity/CF baselines

## 2026-10-26 — Phase 0 Amendments
- reports/eval_YYYYMMDD/ committed (.md/.json), reports/raw/ gitignored — mentor must reproduce
- Interaction sparsity: 80/15/5 split enforced by generator
- Tag vocabulary: 25 tags, 5 domains x 5 tags, 2-4 tags/course, all 25 used, max 30% per tag

## 2026-10-26 — Phase 1
- Fixed seed = 42 for reproducibility; MD5 checksums on all generated artifacts
- Catalog: ~500 courses, IDs course-0001..course-0500, 2-4 tags per course from 25-tag vocabulary
- Users: ~200 users, IDs user-0001..user-0200, tier free/premium, joined_days_ago 0-1000
- Interactions: viewed/completed/rated actions, rating 1-5 only when action=rated, ISO-8601 UTC timestamps
- Challenge set: 20 users designated in splits.json, held out from train/test split
- Train/test: 80/20 user-level split, no user appears in both
- Generator writes to data/ (gitignored); checksums.json enables reproducibility verification

## 2026-10-26 — Phase 2
- Three recommenders: Random (seed-controlled), Popularity (completed-count ranking), UserCF (cosine similarity, K=20 neighbors, cold-start -> popularity)
- Common interface: recommend(user_id, n=10) -> list[str]
- Hard guardrail: completed courses always excluded from recommendations
- Fit only on train users and train interactions; test/challenge data never used in fit
- Fitted state saved via pickle to models/{recommender}.pkl for reuse in evaluation
- Evaluation on test users only (36 users, 34 with completed-course ground truth)
- Metrics: Precision@10, Recall@10, NDCG@10
- Results: random P@10=0.0059, popularity P@10=0.0088, user_cf P@10=0.0088 (CF cold-start falls back to popularity since test users have no train interactions)
- Reports committed to reports/eval_20261026/baselines.md and baselines.json
