# Baselines Evaluation

Date: 20261026

| Recommender | Precision@10 | Recall@10 | NDCG@10 | Users |
|---|---|---|---|---|
| random | 0.0015 | 0.005 | 0.007 | 67 |
| popularity | 0.0015 | 0.005 | 0.0027 | 67 |
| user_cf | 0.0 | 0.0 | 0.0 | 67 |

Ground truth: courses completed by test users in held-out interactions.
Fit only on train users and train interactions.