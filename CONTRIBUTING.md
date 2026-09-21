# Contributing

## Development workflow

1. Create a branch from `main`.
2. Make one coherent change per pull request.
3. Add or update tests for behavioural changes.
4. Run the full local quality suite.
5. Open a pull request. Do not push directly to `main`.

## Local checks

```bash
poetry install
poetry run ruff check .
poetry run mypy src tests
poetry run pytest
```

## Code standards

- Python code must be typed.
- Public functions and classes require docstrings.
- Statistical assumptions should be documented close to the implementation.
- Avoid notebook-only logic for maintained functionality.
- Avoid opaque forecasting wrappers when the underlying model can be expressed directly.
- New numerical methods require tests for edge cases and at least one numerical regression case.

## Pull requests

PR descriptions should explain the problem, approach, statistical implications when relevant, validation performed, and any backward-incompatible behaviour.
