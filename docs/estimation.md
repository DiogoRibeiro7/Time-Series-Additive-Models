# Ordinary least squares

The first estimation layer fits an explicit named design matrix using ordinary least squares (OLS). It deliberately does not add trend, seasonality, an intercept, regularisation, or an error model on behalf of the caller.

## Model

For a design matrix `X` and response vector `y`, the model is

```text
y = X beta + epsilon
```

When `X` has full column rank, OLS estimates the coefficient vector that minimises the residual sum of squares. Coefficients are estimated with `numpy.linalg.lstsq`; the implementation does not explicitly invert `X'X` for fitting.

## Classical covariance

The reported standard errors are the classical homoskedastic OLS standard errors. The residual variance is estimated as

```text
sigma2_hat = RSS / (n - p)
```

and the coefficient covariance matrix is

```text
sigma2_hat * inverse(X'X)
```

These standard errors assume uncorrelated errors with constant variance. That assumption is often inappropriate for time-series data. The covariance is provided as a transparent baseline, not as a default claim that the assumptions hold.

Autocorrelation-aware estimation and robust covariance estimators belong in separate layers.

## Rank and degrees of freedom

`fit_ols` rejects rank-deficient designs. It also requires `n > p`, because the classical residual variance estimate needs positive residual degrees of freedom.

The fitted result reports the numerical condition number of `X`. A large condition number is a warning that coefficient estimates may be unstable even when the matrix is technically full rank.

## Example

```python
from time_series_additive_models.modelling import (
    TimeTransform,
    combine_basis_matrices,
    fit_ols,
    fourier_seasonal_basis,
    polynomial_trend_basis,
)

time = [0.0, 1.0, 2.0, 3.0, 4.0]
response = [1.2, 2.1, 2.9, 4.2, 5.1]

transform = TimeTransform.fit(time)
trend = polynomial_trend_basis(time, degree=1, transform=transform)
seasonal = fourier_seasonal_basis(time, period=4.0, order=1)
design = combine_basis_matrices(trend, seasonal)

result = fit_ols(design, response)

print(result.coefficients)
print(result.standard_errors)
print(result.condition_number)
```

No intercept is added automatically. If the model requires an intercept, the design matrix must contain one.

## Prediction

`OLSResult.predict` accepts a `BasisMatrix` whose column names and ordering exactly match the fitted design. This prevents accidental coefficient/design misalignment.

Prediction currently returns point predictions only. Confidence and prediction intervals will be introduced separately so their assumptions remain explicit.
