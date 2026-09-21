# Time-Series Additive Models

[![CI](https://github.com/DiogoRibeiro7/Time-Series-Additive-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/Time-Series-Additive-Models/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/Time-Series-Additive-Models/actions/workflows/docs.yml/badge.svg)](https://github.com/DiogoRibeiro7/Time-Series-Additive-Models/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A typed Python project for transparent statistical additive modelling of time series.

The project focuses on models whose components and assumptions remain visible: smooth trend terms, seasonal bases, interventions, serially dependent errors, diagnostics, uncertainty, and rolling-origin evaluation. It does not use Prophet in the maintained modelling stack.

## Project status

The repository is under active redevelopment. The engineering foundation is in place, while the maintained statistical model layer is being implemented incrementally and reviewed through pull requests.

The original notebook-based analysis is preserved under `legacy/` for provenance. It is not part of the maintained package or execution path.

## Statistical scope

The core model family starts from an additive representation

```text
y_t = f(t) + s(t) + x_t^T beta + epsilon_t
```

where `f(t)` represents smooth trend structure, `s(t)` periodic or seasonal structure, `x_t` observed regressors and interventions, and `epsilon_t` a residual process that may be serially dependent.

The maintained implementation is intended to cover:

- regression splines for nonlinear trend;
- Fourier and cyclic seasonal bases;
- intervention and structural-break terms;
- GLS and ARIMA-error formulations where appropriate;
- structural time-series and state-space formulations;
- residual and calibration diagnostics;
- rolling-origin evaluation and comparative model assessment;
- uncertainty intervals derived from explicit statistical models.

## Repository layout

```text
src/time_series_additive_models/   maintained Python package
tests/                             unit, integration, and statistical tests
docs/                              methodology and engineering documentation
data/                              data policy and legacy research datasets
legacy/                            historical notebook-based analyses
.github/                           CI, templates, and repository automation
```

Maintained modelling logic belongs under `src/time_series_additive_models/`. New analysis should not be added directly to notebooks when it can be expressed as tested package code.

## Installation

The project uses Poetry.

```bash
git clone https://github.com/DiogoRibeiro7/Time-Series-Additive-Models.git
cd Time-Series-Additive-Models
poetry install
```

The historical datasets use Git LFS. They are not required for package installation, but can be retrieved with:

```bash
git lfs install
git lfs pull
```

## Quality checks

Run the same checks used by CI:

```bash
poetry run ruff check src tests
poetry run mypy src tests
poetry run pytest --cov --cov-report=term-missing
poetry run mkdocs build --strict
```

The repository currently supports Python 3.11 and 3.13 in CI.

## Documentation

Project documentation lives under `docs/` and is built with MkDocs Material.

```bash
poetry run mkdocs serve
```

The documentation covers architecture, methodology, statistical assumptions, and development conventions.

## Legacy analysis

The original analysis used historical Quandl data and `fbprophet`. It is retained solely to preserve the development history of the project.

The notebook is located at:

```text
legacy/notebooks/additive_models_for_prediction.ipynb
```

Associated datasets are stored under `data/legacy/`.

No new maintained code should depend on the legacy notebook or on Prophet.

## Development

Contributions should be made through pull requests against `main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for coding, testing, and review conventions.

Security-sensitive issues should follow [SECURITY.md](SECURITY.md).

## Roadmap

Near-term work is focused on:

1. implementing reusable spline and seasonal basis construction;
2. introducing typed model interfaces and validated input structures;
3. adding residual diagnostics and autocorrelation-aware models;
4. implementing rolling-origin evaluation;
5. adding numerical and statistical regression tests;
6. expanding methodology documentation with reproducible maintained examples.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).
