# Requirements — AI Recommendation Prototype (Capstone, Project 3)

**Status:** Phase 0 — targets defined before code. This file is never edited after approval.  
**Date:** 2026-10-26  
**Owner:** Intern (Kilo-assisted). Stakeholder sign-off required before Phase 1.

## Stakeholder
Primary: learner on the platform ("recommend courses I might take next").  
Secondary: platform operator (catalog coverage, cold-start handling, guardrails).

## Scope
Simulated e-learning catalog recommendation engine:
- ~500 courses (fixed seed, generator script included).
- ~200 synthetic users with interaction histories (viewed, completed, rated 1–5).
- Train/test split on users (80/20); 20-user sparse/unusual history challenge set.
- Recommend top-N courses given a user's interaction history.
- Baselines: popularity, random, user-based collaborative filtering.
- Content-based: TF-IDF on tags + description, cosine similarity.
- Hybrid: weighted blend of collaborative + content-based.
- LLM re-ranking: OpenRouter via OPENAI_API_KEY + `--base-url`, cold-start description embeddings + re-rank top-20 candidates (paid calls require explicit approval before each batch).
- Evaluation: Precision@K, Recall@K, NDCG@K, coverage, diversity; baseline vs each approach; dated evidence files.
- Guardrails: no already-completed courses, cold-start handling, diversity (≤3 from same tag), confidence score.
- Error analysis, responsible-use notes, README, mentor review, slides outline.

## Out of scope
- Production deployment, real user data, live API ingestion.
- Model fine-training, vector databases, A/B testing infrastructure.
- Any paid LLM call without prior approval (LLM used only for embeddings + re-ranking, never as the core scorer).
- Tuning hyperparameters on test or challenge data.

## Catalog schema (course.csv)
| Column | Type | Constraints |
|---|---|---|
| id | string | unique, course-{n} |
| title | string | non-empty |
| tags | string | pipe-separated, ≥1 tag from fixed tag vocabulary |
| difficulty | string | one of: beginner, intermediate, advanced |
| duration_minutes | int | 10–480 |
| description | string | non-empty, ≥50 chars |
| tags | string | Tag vocabulary: exactly 25 distinct tags across 5 domains (technology, business, design, science, personal-development), 5 tags per domain. Each course has 2–4 tags. Generator must use all 25 tags with no tag appearing in more than 30% of courses. |

## User schema (users.csv)
| Column | Type | Constraints |
|---|---|---|
| user_id | string | unique, user-{n} |
| tier | string | one of: free, premium |
| joined_days_ago | int | ≥0 |

## Interaction schema (interactions.csv)
| Column | Type | Constraints |
|---|---|---|
| user_id | string | references users.user_id |
| course_id | string | references catalog.id |
| action | string | one of: viewed, completed, rated |
| rating | int | 1–5, only when action=rated |
| timestamp | string | ISO-8601 UTC |
Distribution: 80% of users have 3–15 interactions (regular); 15% have 16–40 (power); 5% (challenge set) have 0–2 (cold-start/sparse). Generator must enforce this distribution.

## Success metrics (numeric targets, measured on test users @K=10)
| Metric | Target vs random baseline | Target vs popularity baseline | Target vs user-based CF |
|---|---|---|---|
| Precision@10 | ≥ 2.0× | ≥ 1.1× | ≥ 1.0× (hybrid must beat or tie CF) |
| Recall@10 | ≥ 2.0× | ≥ 1.0× | ≥ 1.0× |
| NDCG@10 | ≥ 2.0× | ≥ 1.05× | ≥ 1.0× |
| Coverage (catalog % ever recommended) | — | ≥ 25% | — |
| Diversity (unique tags in top-10) | — | ≥ 6 | — |
| Cold-start Precision@10 | — | ≥ random + 0.05 absolute | — |

Targets are floor targets, not ceilings. Baseline vs each approach reported in dated evidence files.

## Edge cases
1. Cold-start user: no interactions → popularity fallback + description-embedding semantic ranking; confidence = low.
2. Sparse history: 1–2 interactions → content-based on those items; confidence = medium.
3. Already-completed course → excluded from recommendations (hard guardrail).
4. User with all courses completed → return empty list with reason, confidence = 0.
5. Challenge set: 20 users with sparse/unusual histories; evaluated separately, not tuned on.
6. New course (no interactions) → content-based only; never excluded from candidate pool by popularity.

## Acceptance criteria
- [ ] Generator produces identical MD5 checksums across runs with fixed seed.
- [ ] Train/test split is user-level (80/20), no user appears in both.
- [ ] No recommended course appears in the user's completed set.
- [ ] Top-10 contains ≤3 courses sharing any single tag.
- [ ] Every recommendation carries a confidence score (low/medium/high).
- [ ] Metrics computed for Precision@10, Recall@10, NDCG@10 vs all baselines.
- [ ] Evaluation evidence .md and .json files committed to reports/eval_YYYYMMDD/.
- [ ] LLM calls gated behind explicit approval; stub/mock path available for offline runs.
- [ ] pytest suite passes; lint/typecheck clean.
- [ ] README + slides outline + mentor review notes delivered.

## Risks
- Synthetic data may not reflect real learner behavior; metrics are internal validity only.
- LLM embedding/re-ranking cost; gate all calls behind approval, cache embeddings.
- Popularity bias crowding out niche courses; mitigate via diversity guardrail and hybrid blending.
- Cold-start users have no signal; expected lower precision, mitigated with description embeddings.
- Overfitting to test set prohibited; tuning only on train split, challenge set held out.

## Folder tree (proposed)
```
ai-recommendation-prototype/
├── docs/
│   ├── requirements.md   (this file — frozen after approval)
│   ├── work_log.md       (dated entries, append-only)
│   └── decision_log.md   (dated decisions, append-only)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py    (Phase 1: catalog + users + interactions)
│   │   └── loader.py
│   ├── recommend/
│   │   ├── __init__.py
│   │   ├── baselines.py    (Phase 2: popularity, random, user-CF)
│   │   ├── content.py      (Phase 3: TF-IDF + cosine)
│   │   ├── hybrid.py       (Phase 4: weighted blend)
│   │   ├── llm_rerank.py   (Phase 5: embeddings + re-rank, approval-gated)
│   │   └── guardrails.py   (Phase 7: completed-exclusion, diversity, confidence)
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── metrics.py      (Phase 6: P@K, R@K, NDCG@K, coverage, diversity)
│   │   └── report.py       (dated evidence writers)
│   └── __main__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_generator.py
│   ├── test_baselines.py
│   ├── test_content.py
│   ├── test_hybrid.py
│   ├── test_guardrails.py
│   ├── test_metrics.py
│   └── test_challenge.py
├── data/                 (generated artifacts — gitignored)
│   ├── catalog.csv
│   ├── users.csv
│   ├── interactions.csv
│   └── splits.json
├── models/               (saved artifacts — gitignored)
├── reports/              (evaluation evidence)
│   ├── eval_YYYYMMDD/   (.md and .json committed; mentor must reproduce)
│   └── raw/              (gitignored)
├── notebooks/            (optional exploration)
├── requirements.txt      (all deps, listed upfront)
├── pyproject.toml
├── README.md             (Phase 8)
├── slides.md             (Phase 8 outline)
├── .gitignore
└── LICENSE
```

## Phases (in order, one at a time)
1. Data generator (fixed seed, reproducible MD5).
2. Baselines: popularity, random, user-based CF.
3. Content-based: TF-IDF tags + description, cosine similarity.
4. Hybrid: weighted blend.
5. LLM re-ranking (approval-gated).
6. Evaluation: metrics + dated evidence.
7. Guardrails: completed-exclusion, cold-start, diversity, confidence.
8. Error analysis, responsible-use notes, README, mentor review, slides outline.

Commit per phase. PowerShell only. Ask before paid calls or installs.

Amended 2026-10-26: reports gitignore, interaction sparsity, tag vocabulary (pre-Phase-1 review).

---
*End of Phase 0. Waiting for "go" to start Phase 1.*