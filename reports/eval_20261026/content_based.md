# Content-Based Evaluation

Date: 20261026

Method: TF-IDF on tags (pipe-separated, 0.7 weight) + TF-IDF on description (0.3 weight). Concatenated sparse features. User profile = weighted average of interacted course vectors (completed=1.0, rated>=4=0.8, rated 1-3=0.3, viewed=0.1). Cosine similarity, exclude train-completed courses. Cold-start falls back to popularity.
Ground truth: completed courses + courses rated >= 4 by test users in held-out interactions.
Fit only on train users and train interactions.

| Recommender | K | Precision@K | Recall@K | NDCG@K | Hit Rate@K | Users |
|---|---|---|---|---|---|---|
| content_based | 5 | 0.0047 | 0.0176 | 0.0087 | 0.0235 | 85 |
| content_based | 10 | 0.0024 | 0.0176 | 0.0087 | 0.0235 | 85 |
| content_based | 20 | 0.0029 | 0.0471 | 0.0163 | 0.0588 | 85 |