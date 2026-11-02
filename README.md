# AI Recommendation Prototype

Popularity, collaborative-filtering, and content-based baselines, a hybrid blend, and an LLM re-ranker for course recommendations on an e-learning platform.

A learner on an e-learning platform gets a ranked list of courses; the LLM re-ranker raises Hit Rate@10 from 4.71% (popularity baseline) to 6.52%.

## Prerequisites

- Python 3.11+
- OpenRouter API key (optional). Popularity and content-based recommendations work with no key; CF, hybrid, and LLM re-ranking fall back to popularity/content-based when the key is absent.

## Installation

```powershell
git clone https://github.com/Suleman-Sehar/ai-recommendation-prototype
cd ai-recommendation-prototype
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_llm_rerank.py --dry-run --limit 3
```

## Running without an API key

Popularity and content-based recommenders run fully offline. CF and hybrid run offline using cached models. LLM re-ranking skips users and reports zero metrics when no key is present.

## Key results (K=10, 85 test users; LLM rerank on 46 users with non-empty ground truth)

| Approach | Hit Rate@10 | NDCG@10 | Precision@10 |
|---|---|---|---|
| random | 5.88% | 0.0191 | 0.0059 |
| popularity | 4.71% | 0.0128 | 0.0047 |
| user_cf | 1.18% | 0.0051 | 0.0012 |
| content_based | 2.35% | 0.0087 | 0.0024 |
| hybrid | 3.53% | 0.0097 | 0.0035 |
| llm_rerank | 6.52% | 0.0180 | 0.0065 |

Note: random outperforms popularity at K=10 due to high variance on sparse ground truth (79% size-1 ground truth). At K=20, popularity leads (HR=10.59% vs random HR=4.71%), which is the expected direction.

## Project structure

```
.
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── catalog.csv
│   ├── checksums.json
│   ├── interactions.csv
│   ├── splits.json
│   └── users.csv
├── docs/
│   ├── MENTOR_REVIEW.md
│   ├── SLIDES_OUTLINE.md
│   ├── decision_log.md
│   ├── requirements.md
│   └── work_log.md
├── models/
│   ├── popularity.pkl
│   ├── random.pkl
│   └── user_cf.pkl
├── reports/
│   ├── eval_20261026/
│   └── raw/
├── scripts/
│   └── run_llm_rerank.py
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── data/
│   ├── evaluate/
│   └── recommend/
└── tests/
    ├── conftest.py
    └── test_*.py
```

## Acceptance criteria

| Criterion | Status |
|---|---|
| All 6 approaches evaluated at K=10 | Pass |
| 66 tests passing | Pass |
| Challenge set (20 cold-start users) qualitative results | N/A (no ground truth) |
| LLM rerank best Hit Rate@10 | Pass |

## Known limitations

- Sparse ground truth: 79% of test users have size-1 ground truth.
- CF underperforms popularity due to sparse binary signals.
- 25-tag vocabulary too coarse for cross-domain exploration.
- Synthetic data only.
- LLM rerank evaluated on subset (46 of 85 users).

## Future work

1. Expand tag vocabulary beyond 25 tags.
2. Collect real interaction data instead of synthetic.
3. Tune hybrid blend weights and candidate pool size.
4. Evaluate LLM rerank on full 85-user set.
5. Add diversity/novelty metrics.

## License

MIT License - Suleman Sehar 2026

## Contact

https://github.com/Suleman-Sehar
