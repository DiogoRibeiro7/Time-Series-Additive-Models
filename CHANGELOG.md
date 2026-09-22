# Changelog

All notable changes to this project will be documented in this file.

The format follows the principles of Keep a Changelog. The project is currently pre-release.

## [Unreleased]

### Added

- residual autocorrelation estimation and Ljung-Box portmanteau diagnostics;
- residual mean, standard deviation, RMSE, and MAE summaries;
- explicit polynomial, piecewise-linear, and Fourier basis construction;
- reusable affine time transformation for consistent training and future design matrices;
- immutable named basis matrices and safe basis composition;
- installable typed Python package structure;
- Poetry-based development environment;
- Ruff, mypy, pytest, coverage, and pre-commit checks;
- GitHub Actions CI across supported Python versions;
- MkDocs documentation and strict documentation builds;
- contribution, security, pull-request, and issue templates;
- citation metadata and repository-level documentation.

### Changed

- maintained modelling direction moved away from Prophet toward explicit statistical additive models;
- historical notebook and datasets moved into clearly labelled legacy locations.

### Security

- removed the hard-coded Quandl API key from the historical notebook.
