# Architecture

## Repository layers

```text
src/time_series_additive_models/
    modelling/       statistical model components
    diagnostics/     residual and calibration diagnostics
    evaluation/      rolling-origin and comparative evaluation
    data/            validated data-loading utilities

tests/
    unit/            deterministic unit tests
    integration/     end-to-end workflow tests
    statistical/     numerical and statistical regression checks

docs/
    methodology and engineering documentation

legacy/
    historical analyses retained for provenance
```

## Design principles

The maintained code should expose assumptions explicitly. Forecasting behaviour must be inspectable through statistical components rather than hidden behind a single high-level forecasting API.

Numerical code should be deterministic where possible. Randomized procedures must accept an explicit random seed. Public functions should be typed, documented, and tested.

The modelling layer should remain independent from plotting and notebook presentation code.
