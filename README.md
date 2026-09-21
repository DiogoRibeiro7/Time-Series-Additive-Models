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

The notebook predates the current Python Prophet API and the current Nasdaq Data Link client. It should therefore be treated as historical analysis until its dependencies and data-access code have been migrated.

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
2. migrate the notebook from `fbprophet` to the current `prophet` package;
3. migrate legacy Quandl access to the current Nasdaq Data Link client or replace remote retrieval with reproducible local inputs;
4. extract reusable preprocessing and modelling code from the notebook;
5. add statistical tests and regression checks for the extracted code;
6. separate historical demonstrations from maintained examples;
7. document assumptions, data provenance, model limitations, and reproducibility.

## Scientific perspective

Additive forecasting models are useful when a time series can be represented through components such as a smooth trend, periodic structure, known events, and residual variation. Their convenience does not remove the need to examine identifiability, structural breaks, extrapolation assumptions, parameter sensitivity, and forecast uncertainty.

The purpose of this repository is therefore not to present Prophet as a universal forecasting method. It is to use additive models as an interpretable modelling framework and to make the assumptions behind the forecasts explicit.

## License

Apache License 2.0. See `LICENSE`.
