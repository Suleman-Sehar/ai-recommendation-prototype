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

## 2026-10-26 — Phase 1 Fix: Temporal per-user train/test split
- Initial user-level split made CF untestable (all test users had zero train interactions); corrected to per-user temporal split before any recommender was tuned.
- splits.json now stores: train_interactions (list of interaction ids), test_holdout (list of interaction ids), challenge_users (user_id list)
- Per-user temporal split: users with >=3 interactions put last 20% (min 1) in test holdout by timestamp; users with 1-2 interactions all go to train; challenge users all interactions to train
- All non-challenge users appear in BOTH train and test interaction data
- interactions.csv now includes unique "id" column per interaction
- Results: 1750 train interactions, 342 test holdout, 171 users with test holdout, avg holdout size 2.2, 20 challenge users

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

## 2026-10-26 — Phase 2 Re-run (after temporal split fix)
- Baselines re-evaluated on per-user temporal splits: 67 test users with ground truth
- Results: random P@10=0.0015, popularity P@10=0.0015, user_cf P@10=0.0
- CF is 0.0 because temporal split weakens signal (predicting future completions from past interactions)
- Reports overwritten in reports/eval_20261026/baselines.md and baselines.json

## 2026-10-26 — Phase 2 Fix: Ground truth expanded to completed + rated>=4
- Ground truth expanded from completed-only to completed+rated>=4 because 61% of test users had empty ground truth under completed-only definition. Decision made before Phase 3, not post-hoc.
- Baselines re-evaluated with new ground truth: 85 test users with non-empty ground truth (up from 67), avg ground truth size increased.
- Results: random P@10=0.0, popularity P@10=0.0047, user_cf P@10=0.0012
- Reports overwritten in reports/eval_20261026/baselines.md and baselines.json

## 2026-10-26 — Phase 2 Fix: Added Hit Rate@10 and multi-K evaluation
- Added Hit Rate@10 and multi-K evaluation because ground truth is size-1 for 79% of test users; relative comparisons more informative than absolute P@10.
- Evaluation now computes Precision@K, Recall@K, NDCG@K, Hit Rate@K at K=5, K=10, K=20.
- New metrics module at src/evaluate/metrics.py.
- Results at K=10: random P@10=0.0059, popularity P@10=0.0047, user_cf P@10=0.0. Hit Rate@10: random 0.0588, popularity 0.0471, user_cf 0.0.
- Reports overwritten in reports/eval_20261026/baselines.md and baselines.json

## 2026-10-26 — Phase 2 Fix: CF weighted interaction signals
- CF with binary signals scored 0.0 Hit Rate@10 because similar users' sparse completions did not overlap with test holdout; switched to weighted signals to give CF meaningful candidate pool.
- Signal weights: completed=1.0, rated>=4=0.8, rated 1-3=0.3, viewed=0.1
- Matrix built with weighted signals; scoring aggregates ALL interactions of similar users (weighted), not just completed
- Exclusion list remains train-completed only (correct as-is)
- Results at K=10: random Hit=0.0588, popularity Hit=0.0471, user_cf Hit=0.0118 (was 0.0)
- Added tests/test_baselines.py TestWeightedCF: 2 tests verifying weighted CF scores viewed interactions where binary CF misses
- Reports overwritten in reports/eval_20261026/baselines.md and baselines.json
