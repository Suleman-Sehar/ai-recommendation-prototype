from __future__ import annotations

from collections import Counter
from typing import List

import pandas as pd


def check_diversity(recommendations: List[str], catalog_df, max_per_tag: int = 3) -> List[str]:
    if isinstance(catalog_df, list):
        cat_map = {c['id']: c for c in catalog_df}
    elif isinstance(catalog_df, pd.DataFrame):
        cat_map = {str(row['id']): row for _, row in catalog_df.iterrows()}
    else:
        cat_map = {}

    result = []
    tag_count = Counter()
    for cid in recommendations:
        c = cat_map.get(cid) or cat_map.get(str(cid))
        if c is None:
            result.append(cid)
            continue
        tags = str(c.get('tags', '')).split('|')
        tags = [t.strip() for t in tags if t.strip()]
        would_exceed = False
        for t in tags:
            if tag_count.get(t, 0) >= max_per_tag:
                would_exceed = True
                break
        if would_exceed:
            continue
        result.append(cid)
        for t in tags:
            tag_count[t] += 1
    return result


def assign_confidence(user_id: str, recommendations: List[str], user_history: List[dict] | int, source: str, diversity_triggered: bool = False) -> str:
    if isinstance(user_history, int):
        hist_count = user_history
    elif isinstance(user_history, list):
        hist_count = len(user_history)
    else:
        hist_count = 0

    if hist_count == 0 or diversity_triggered:
        return 'low'
    if source == 'llm_rerank' and hist_count >= 3 and not diversity_triggered:
        return 'high'
    if source in ['hybrid', 'content_based'] and hist_count >= 1 and not diversity_triggered:
        return 'medium'
    if diversity_triggered:
        return 'low'
    return 'medium'

