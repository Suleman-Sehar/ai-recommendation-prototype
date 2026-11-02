# Mentor Review

Repo: [placeholder]

## Summary

- Recommends courses to e-learning learners using popularity, CF, and content-based baselines plus a hybrid blend and an LLM re-ranker.
- LLM re-ranker achieves best Hit Rate@10 of 6.52% vs 4.71% popularity baseline.
- 500 courses, 200 users, temporal per-user split; 85 test users with ground truth.
- 66 tests passing; all evidence committed under reports/eval_20261026/.
- Synthetic data only; LLM rerank evaluated on 46 of 85 users.

## Results (K=10)

| Approach | HR@10 | NDCG@10 | P@10 | Users |
|---|---|---|---|---|
| random | 5.88% | 0.0191 | 0.0059 | 85 |
| popularity | 4.71% | 0.0128 | 0.0047 | 85 |
| user_cf | 1.18% | 0.0051 | 0.0012 | 85 |
| content_based | 2.35% | 0.0087 | 0.0024 | 85 |
| hybrid | 3.53% | 0.0097 | 0.0035 | 85 |
| llm_rerank | 6.52% | 0.0180 | 0.0065 | 46 |

## Limitations

- Sparse ground truth (79% size-1); CF underperforms popularity; 25-tag vocabulary too coarse; synthetic data; LLM rerank on subset.

## Mentor questions

1. Is the 25-tag vocabulary sufficient, or should we expand to free-text descriptions?
2. Should hybrid blend weights be tuned via grid search given candidate-pool union drives gains?
3. How to fairly compare LLM rerank when only 46 of 85 users have ground truth?

## Feedback

<!-- blank -->
