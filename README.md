# Time-Series Additive Models

A research repository for studying additive models in time-series forecasting, with an emphasis on interpretable trend, seasonality, changepoints, and uncertainty.

The repository began as a notebook-based analysis using historical financial and macroeconomic examples. It is now being rehabilitated into a reproducible research project while preserving the original notebook as a record of the earlier work.

## Scope

The original notebook explores several questions:

- comparison of General Motors and Tesla market capitalisation;
- additive forecasting with Prophet;
- sensitivity to the changepoint prior scale;
- forecast uncertainty and trend decomposition;
- long-horizon GDP forecasts for the United States and China.

The notebook is retained as historical material. Its use of `fbprophet` is not part of the maintained modelling direction of this repository.

## Repository status

Modernisation is in progress. The current work deliberately separates infrastructure changes from scientific changes so that numerical and modelling behaviour can be reviewed independently.

The original notebook remains at:

```text
Additive Models for Prediction.ipynb
```

Data files are stored under `data/`. The existing CSV and Excel files were committed through Git LFS, so Git LFS is required when cloning the historical data exactly as stored.

## Development setup

The project uses Poetry for development tooling.

```bash
poetry install
poetry run ruff check .
poetry run mypy tests
poetry run pytest
```

To install Git LFS before pulling the historical data:

```bash
git lfs install
git lfs pull
```

## Credentials

Do not place API keys in notebooks or source files.

For Nasdaq Data Link credentials, use an environment variable:

```bash
export NASDAQ_DATA_LINK_API_KEY="your-key"
```

A local `.env` file may also be used by development tooling, but it is ignored by Git. The committed `.env.example` contains only the variable name and no credential.

## Modernisation roadmap

The rehabilitation is intentionally incremental:

1. establish repository metadata, automated checks, and reproducibility safeguards;
2. retain the original Prophet notebook only as a historical artefact and remove it from the maintained execution path;
3. replace the forecasting implementation with explicit statistical additive models based on spline trends, seasonal bases, interventions, and appropriate error structures;
4. migrate legacy remote-data retrieval to reproducible local inputs or a documented current source;
5. extract reusable preprocessing, fitting, prediction, and diagnostic code into a typed Python package;
6. add unit tests, numerical regression tests, and statistical validation tests;
7. implement rolling-origin evaluation and principled model comparison;
8. document assumptions, data provenance, diagnostics, limitations, and reproducibility.

## Scientific perspective

Additive forecasting models are useful when a time series can be represented through components such as a smooth trend, periodic structure, known events, and residual variation. Their convenience does not remove the need to examine identifiability, structural breaks, extrapolation assumptions, parameter sensitivity, and forecast uncertainty.

The maintained implementation will use additive models as an explicit statistical framework rather than as a wrapper around Prophet. The preferred components are regression splines for nonlinear trend, Fourier or cyclic bases for periodic structure, intervention terms for structural changes, and GLS, ARIMA-error, or state-space formulations when residual dependence remains. Forecast uncertainty and model comparison should come from the fitted statistical model, residual diagnostics, and rolling-origin evaluation rather than from a single forecasting interface.

## License

Apache License 2.0. See `LICENSE`.
