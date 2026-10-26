# Work Log — AI Recommendation Prototype

## 2026-10-26 — Phase 0: Requirements
- Proposed folder tree in docs/requirements.md
- Defined stakeholder, scope/out-of-scope, schemas, success metrics, edge cases, acceptance criteria, risks
- File frozen — no edits after approval

## 2026-10-26 — Phase 0 Amendments (pre-Phase-1 review)
- Updated reports gitignore policy: reports/eval_YYYYMMDD/ (.md/.json committed), reports/raw/ gitignored
- Added interaction sparsity distribution: 80% regular (3-15), 15% power (16-40), 5% challenge (0-2)
- Added tag vocabulary: 25 tags across 5 domains, 5 per domain, 2-4 tags per course, all 25 used, no tag >30% of courses
- Added change note: "Amended 2026-10-26: reports gitignore, interaction sparsity, tag vocabulary (pre-Phase-1 review)."

## 2026-10-26 — Phase 1: Data Generator
- Created project structure: src/{data,recommend,evaluate}/, tests/, data/, reports/raw/, docs/
- Wrote requirements.txt (pandas, scikit-learn, numpy, sentence-transformers, pytest, pytest-cov)
- Wrote pyproject.toml, .gitignore, src/config.py, src/__main__.py, src/data/__init__.py, src/data/loader.py
- Wrote src/data/generator.py: fixed-seed catalog (~500 courses), users (~200), interactions with sparsity distribution
- Tag vocabulary: 25 tags across 5 domains (technology, business, design, science, personal-development), 5 tags per domain
- Each course: 2-4 tags, all 25 tags used, no tag in >30% of courses
- Interaction distribution: 80% regular (3-15), 15% power (16-40), 5% sparse/challenge (0-2)
- Train/test split on users (80/20), 20 challenge users held out
- Writes catalog.csv, users.csv, interactions.csv, splits.json, checksums.json (MD5)
- Wrote tests/test_generator.py: 30 tests covering schema, constraints, distribution, splits, reproducibility
- All 30 tests pass

## 2026-10-26 — Phase 1 Fix: Temporal per-user train/test split
- Rewrote generator to use per-user temporal split instead of user-level split
- interactions.csv now includes unique "id" column per interaction
- splits.json stores: train_interactions (list of ids), test_holdout (list of ids), challenge_users (user_id list)
- Users with >=3 interactions: last 20% (min 1) by timestamp go to test holdout, rest to train
- Users with 1-2 interactions: all to train (no test ground truth)
- Challenge users: all interactions to train (cold-start, evaluated separately)
- All non-challenge users appear in BOTH train and test interaction data
- Wrote tests/test_generator.py: 32 tests covering schema, constraints, distribution, temporal split, reproducibility
- All 32 tests pass
- Results: 1750 train interactions, 342 test holdout, 171 users with test holdout, avg holdout size 2.2, 20 challenge users

## 2026-10-26 — Phase 2: Baselines
- Created src/recommend/__init__.py and src/recommend/baselines.py
- Implemented RandomRecommender (seed-controlled, excludes completed), PopularityRecommender (ranks by completed count across train users), UserCFRecommender (cosine similarity on user-item matrix, K=20 neighbors, cold-start falls back to popularity)
- Common interface: recommend(user_id, n=10) -> list of course_ids
- Hard guardrail: completed courses always excluded
- Fit only on train users and train interactions (derived from train_interactions split)
- Each recommender saves fitted state via pickle to models/
- Re-evaluated on new temporal splits: 67 test users with ground truth
- Results: random P@10=0.0015, popularity P@10=0.0015, user_cf P@10=0.0
- Saved reports/eval_20261026/baselines.md and baselines.json (committed)
- Wrote tests/test_baselines.py: 11 tests covering interface, completed-exclusion, random reproducibility, popularity ranking, CF cold-start fallback, CF completed-exclusion, evaluation structure
- All 11 tests pass
