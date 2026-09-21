# Methodology

The maintained model family is additive:

```text
y_t = f(t) + s(t) + x_t^T beta + epsilon_t
```

where:

- `f(t)` is a smooth trend;
- `s(t)` captures periodic or seasonal structure;
- `x_t` contains observed regressors and interventions;
- `epsilon_t` is a residual process that may be serially dependent.

Candidate implementations include regression splines, cyclic or Fourier bases, intervention terms, GLS or ARIMA-error models, and state-space formulations.

Model selection should combine diagnostics with rolling-origin evaluation. Residual autocorrelation, heteroskedasticity, structural instability, and interval calibration should be examined explicitly.
