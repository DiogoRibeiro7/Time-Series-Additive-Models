# Basis construction

The first maintained modelling layer provides explicit design-matrix components for additive time-series models. These functions construct predictors only; they do not fit coefficients or hide an estimation procedure.

## Reusable time scaling

Polynomial and piecewise-linear trend bases use an affine `TimeTransform`. Fitting the transform on the training interval and reusing it for future observations prevents the training and prediction matrices from being defined on different scales.

```python
from time_series_additive_models.modelling import TimeTransform

transform = TimeTransform.fit([0.0, 1.0, 2.0, 3.0])
future = transform.transform([4.0, 5.0])
```

The fitted training interval maps to `[0, 1]`; future observations may legitimately lie outside that interval.

## Polynomial trend

`polynomial_trend_basis` constructs powers of scaled time. For degree two the columns are

```text
1, t, t^2
```

on the transformed time coordinate. Scaling does not resolve all conditioning problems of high-degree polynomials, so low degrees should be preferred and spline-based components should be used for more flexible trends.

## Piecewise-linear trend

`piecewise_linear_trend_basis` implements continuous hinge functions. With a knot `k`, an additional column is

```text
max(0, t - k)
```

after applying the fitted time transform. This produces an explicit continuous piecewise-linear trend and makes slope changes identifiable through ordinary regression coefficients.

Knots are specified in the original time units and must lie strictly inside the fitted training interval.

## Fourier seasonality

`fourier_seasonal_basis` creates sine/cosine pairs

```text
sin(2 pi h t / P), cos(2 pi h t / P)
```

for harmonics `h = 1, ..., H`, where `P` is the period. Time and period must use the same units. The function supports irregularly spaced time coordinates, but the scientific meaning of the chosen period remains the caller's responsibility.

## Combining components

`combine_basis_matrices` concatenates named basis matrices after checking row compatibility and duplicate column names.

```python
from time_series_additive_models.modelling import (
    TimeTransform,
    combine_basis_matrices,
    fourier_seasonal_basis,
    piecewise_linear_trend_basis,
)

time = [0.0, 1.0, 2.0, 3.0, 4.0]
transform = TimeTransform.fit(time)

trend = piecewise_linear_trend_basis(
    time,
    knots=[2.0],
    transform=transform,
)
seasonal = fourier_seasonal_basis(
    time,
    period=4.0,
    order=1,
)

design = combine_basis_matrices(trend, seasonal)
```

The resulting `BasisMatrix` stores an immutable numeric matrix together with explicit column names. Estimation, regularisation, diagnostics, and uncertainty are intentionally separate layers.
