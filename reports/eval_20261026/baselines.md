# Baselines Evaluation

Date: 20261026

| Recommender | Precision@10 | Recall@10 | NDCG@10 | Users |
|---|---|---|---|---|
| random | 0.0059 | 0.0135 | 0.0073 | 34 |
| popularity | 0.0088 | 0.0265 | 0.019 | 34 |
| user_cf | 0.0088 | 0.0265 | 0.019 | 34 |

Ground truth: courses completed by test users in held-out interactions.
Fit only on train users and train interactions.