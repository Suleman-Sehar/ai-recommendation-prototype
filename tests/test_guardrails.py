from __future__ import annotations

from src.data.loader import load_catalog
from src.recommend.guardrails import check_diversity, assign_confidence


class TestGuardrails:
    def test_diversity_check_reduces_same_tag(self):
        catalog = load_catalog()
        # Find courses with same tag
        cat_map = {c['id']: c for c in catalog}
        # Get all courses sharing a common tag
        tag = 'javascript'
        same_tag = [c['id'] for c in catalog if tag in str(c.get('tags','')).split('|')]
        if len(same_tag) >= 4:
            recs = same_tag[:5]
            result = check_diversity(recs, catalog, max_per_tag=3)
            assert len(result) <= 3

    def test_confidence_high(self):
        conf = assign_confidence('u1', ['c1','c2'], [{'x':1},{'y':2},{'z':3}], source='llm_rerank', diversity_triggered=False)
        assert conf == 'high'

    def test_confidence_medium(self):
        conf = assign_confidence('u1', ['c1','c2'], [{'x':1}], source='hybrid', diversity_triggered=False)
        assert conf == 'medium'

    def test_confidence_low_cold_start(self):
        conf = assign_confidence('u1', ['c1'], [], source='hybrid', diversity_triggered=False)
        assert conf == 'low'

    def test_confidence_low_diversity_triggered(self):
        conf = assign_confidence('u1', ['c1'], [{'x':1},{'y':2},{'z':3}], source='llm_rerank', diversity_triggered=True)
        assert conf == 'low'

