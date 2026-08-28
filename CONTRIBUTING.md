# Contributing to GenRec

Thanks for helping improve GenRec. Keep changes focused so experiments remain
reproducible and reviewable.

## Development setup

Create a Python 3.10 virtual environment, then install the development tools:

```bash
python -m pip install -r requirements-dev.txt
```

The full dependency set includes GPU-oriented packages. CPU-only contributors
can install `pytest` by itself when working only on repository checks or docs.

## Before opening a pull request

Run the fast checks from the repository root:

```bash
python -m compileall -q scripts tests
python -m pytest -q
python scripts/validate_repository.py
```

Do not commit model checkpoints, generated outputs, credentials, or local data
paths. Large artifacts that belong in the project must use the existing Git LFS
rules.

## Commit style

Use a short, imperative summary that explains the change, for example:

```text
clarify cpu-only installation steps
add validation for preference data
```

Keep unrelated code, data, and documentation changes in separate commits.
