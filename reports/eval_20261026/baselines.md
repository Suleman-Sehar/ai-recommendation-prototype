# Baselines Evaluation

Date: 20261026

| Recommender | Precision@10 | Recall@10 | NDCG@10 | Users |
|---|---|---|---|---|
| random | 0.0 | 0.0 | 0.0 | 85 |
| popularity | 0.0047 | 0.0333 | 0.0128 | 85 |
| user_cf | 0.0012 | 0.0118 | 0.0034 | 85 |

Ground truth: completed courses + courses rated >= 4 by test users in held-out interactions.
Fit only on train users and train interactions.