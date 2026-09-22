# Robust covariance estimation

Robust covariance estimation changes inference while leaving the OLS coefficient vector unchanged.

## Heteroskedasticity-consistent covariance

The project exposes HC0, HC1, HC2, and HC3 covariance estimators.

- HC0 uses squared residuals directly.
- HC1 applies the finite-sample factor `n / (n - p)`.
- HC2 divides each squared residual by `1 - h_ii`.
- HC3 divides each squared residual by `(1 - h_ii)^2`.

Here `h_ii` denotes diagonal leverage from the OLS hat matrix.

The sandwich form is

```text
(X'X)^-1 X' Omega X (X'X)^-1
```

with the diagonal structure of `Omega` determined by the selected HC estimator.

## Newey-West HAC covariance

For serially correlated and heteroskedastic errors, `newey_west_covariance` uses Bartlett kernel weights up to an explicit maximum lag.

The long-run covariance estimate adds weighted lagged score cross-products:

```text
w_l = 1 - l / (L + 1)
```

for lags `l = 1, ..., L`.

The optional small-sample correction multiplies the meat matrix by `n / (n - p)`.

## Interpretation

These estimators alter standard errors and covariance matrices only. They do not correct biased coefficient estimates arising from omitted dynamics, endogeneity, misspecified regressors, or an inappropriate conditional-mean model.

Likewise, HAC covariance is not a substitute for modelling serial structure when the serial dependence is itself scientifically meaningful.

## Example

```python
from time_series_additive_models.modelling import (
    heteroskedasticity_consistent_covariance,
    newey_west_covariance,
)

hc3 = heteroskedasticity_consistent_covariance(
    result,
    design,
    kind="HC3",
)

hac = newey_west_covariance(
    result,
    design,
    max_lag=4,
)

print(hc3.standard_errors)
print(hac.standard_errors)
```

The choice of covariance estimator should follow the error structure and inferential goal rather than convenience.
