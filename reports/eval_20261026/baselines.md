# Baselines Evaluation

Date: 20261026

Ground truth: completed courses + courses rated >= 4 by test users in held-out interactions.
Fit only on train users and train interactions.

| Recommender | K | Precision@K | Recall@K | NDCG@K | Hit Rate@K | Users |
|---|---|---|---|---|---|---|
| random | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 85 |
| random | 10 | 0.0059 | 0.0451 | 0.0191 | 0.0588 | 85 |
| random | 20 | 0.0024 | 0.0471 | 0.0169 | 0.0471 | 85 |
| popularity | 5 | 0.0047 | 0.0098 | 0.0057 | 0.0235 | 85 |
| popularity | 10 | 0.0047 | 0.0333 | 0.0128 | 0.0471 | 85 |
| popularity | 20 | 0.0059 | 0.0725 | 0.024 | 0.1059 | 85 |
| user_cf | 5 | 0.0024 | 0.0118 | 0.0051 | 0.0118 | 85 |
| user_cf | 10 | 0.0012 | 0.0118 | 0.0051 | 0.0118 | 85 |
| user_cf | 20 | 0.0024 | 0.0382 | 0.0122 | 0.0471 | 85 |

Metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K (fraction of users with >=1 hit in top-K).