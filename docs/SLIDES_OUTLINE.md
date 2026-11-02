# Slides Outline

## Slide 1: Problem / stakeholder
- E-learning learner needs ranked course recommendations
- Goal: improve discovery beyond popularity
- Baseline popularity HR@10 = 4.71%
- Stakeholder: learner on e-learning platform
- Measurable: Hit Rate@10

Speaker notes: We target a learner who needs relevant next courses. Popularity is the baseline to beat.

## Slide 2: Pipeline / approach
- Data: 500 courses, 200 users, temporal split
- 85 test users with non-empty ground truth
- 20 challenge cold-start users
- Baselines: random, popularity, user CF, content-based
- Hybrid blend + LLM re-ranker
- Guardrails and confidence scoring

Speaker notes: Temporal per-user split avoids leakage. Models trained on history, evaluated on future interactions.

## Slide 3: Baseline comparison
- random HR@10 = 5.88%
- popularity HR@10 = 4.71%
- user_cf HR@10 = 1.18%
- content_based HR@10 = 2.35%
- hybrid HR@10 = 3.53%
- llm_rerank HR@10 = 6.52% (best)

Speaker notes: CF underperforms popularity due to sparse binary signals. Hybrid gains come from candidate pool union.

## Slide 4: LLM re-ranking results
- LLM rerank HR@10 = 6.52% (46 users)
- NDCG@10 = 0.0180
- Precision@10 = 0.0065
- Best approach overall
- Caching and hallucination detection
- Cold-start handling

Speaker notes: LLM re-ranker is clear winner; evaluated on subset with non-empty ground truth.

## Slide 5: Guardrails and cold-start
- Confidence scoring on predictions
- Guardrails filter unsafe/invalid items
- 20 challenge cold-start users evaluated qualitatively
- No ground truth for challenge set
- Graceful fallback without API key

Speaker notes: Cold-start handled via content/popularity fallbacks; challenge set qualitative only.

## Slide 6: Error analysis
- 79% of test users have size-1 ground truth
- Sparse signals hurt CF
- 25-tag vocabulary too coarse
- Cross-domain exploration weak
- Synthetic data limits generalization

Speaker notes: Most errors stem from sparse ground truth and coarse tags.

## Slide 7: Decisions and trade-offs
- Candidate pool union drives hybrid gains, not blend weights
- LLM rerank cost vs accuracy
- Offline popularity/content-based for no-key runs
- Temporal split for honest evaluation
- Synthetic data for reproducibility

Speaker notes: Chose union over weighting; LLM cost acceptable for quality gain.

## Slide 8: Next steps / 10x scale
- Expand tag vocabulary
- Collect real interaction data
- Tune hybrid weights
- Full 85-user LLM rerank eval
- Add diversity/novelty metrics

Speaker notes: Next focus on data quality and scaling evaluation coverage.
