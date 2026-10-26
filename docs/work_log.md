# Work Log — AI Recommendation Prototype

## 2026-10-26 — Phase 0: Requirements
- Proposed folder tree in docs/requirements.md
- Defined stakeholder, scope/out-of-scope, schemas, success metrics, edge cases, acceptance criteria, risks
- File frozen — no edits after approval

## 2026-10-26 — Phase 0 Amendments (pre-Phase-1 review)
- Updated reports gitignore policy: reports/eval_YYYYMMDD/ (.md/.json committed), reports/raw/ gitignored
- Added interaction sparsity distribution: 80% regular (3-15), 15% power (16-40), 5% challenge (0-2)
- Added tag vocabulary: 25 tags across 5 domains, 5 per domain, 2-4 tags per course, all 25 used, no tag >30% of courses
- Added change note: "Amended 2026-10-26: reports gitignore, interaction sparsity, tag vocabulary (pre-Phase-1 review)."

## 2026-10-26 — Phase 1: Data Generator
- Created project structure: src/{data,recommend,evaluate}/, tests/, data/, reports/raw/, docs/
- Wrote requirements.txt (pandas, scikit-learn, numpy, sentence-transformers, pytest, pytest-cov)
- Wrote pyproject.toml, .gitignore, src/config.py, src/__main__.py, src/data/__init__.py, src/data/loader.py
- Wrote src/data/generator.py: fixed-seed catalog (~500 courses), users (~200), interactions with sparsity distribution
- Tag vocabulary: 25 tags across 5 domains (technology, business, design, science, personal-development), 5 tags per domain
- Each course: 2-4 tags, all 25 tags used, no tag in >30% of courses
- Interaction distribution: 80% regular (3-15), 15% power (16-40), 5% sparse/challenge (0-2)
- Train/test split on users (80/20), 20 challenge users held out
- Writes catalog.csv, users.csv, interactions.csv, splits.json, checksums.json (MD5)
- Wrote tests/test_generator.py: 30 tests covering schema, constraints, distribution, splits, reproducibility
- All 30 tests pass
