# GovCan Plain Language Converter

## Testing (Audit / CI)

Unit tests run automatically on every push via GitHub Actions (see `.github/workflows/ci.yml`).

Run tests locally:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

What tests cover:
- Deterministic readability scoring (Unicode-safe).
- Sentence splitting behavior.

Note: Grade ≤ 8 enforcement in the app is enforced at save/export time and depends on LLM outputs; unit tests validate the grading functions used for that enforcement and auditor review.

Auto-deploy test commit
