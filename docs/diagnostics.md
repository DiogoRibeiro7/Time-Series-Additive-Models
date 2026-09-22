# Residual diagnostics

Residual diagnostics are kept separate from estimation. Any fitted model that can provide a residual sequence can use this layer.

## Autocorrelation

The sample autocorrelation function is computed directly from the residual sequence. By default the residual mean is removed before the lagged products are calculated.

For lag k, the implementation uses the biased diagnostic form

```text
rho_k = sum((e_t - e_bar)(e_{t-k} - e_bar)) / sum((e_t - e_bar)^2)
```

with lag zero fixed at one. A zero-variance sequence is rejected because its autocorrelation is undefined.

## Ljung-Box test

The Ljung-Box statistic is computed as

```text
Q = n(n + 2) * sum(rho_k^2 / (n - k))
```

for lags from one through the requested maximum lag. The p-value is obtained from a chi-square reference distribution using SciPy.

The optional `model_df` argument adjusts the reference degrees of freedom:

```text
df = lags - model_df
```

This adjustment is useful when residual autocorrelation parameters have already been estimated. The adjusted degrees of freedom must remain positive.

## Scale summaries

`summarize_residuals` reports:

- residual mean;
- sample standard deviation;
- root mean squared error;
- mean absolute error;
- autocorrelation vector;
- Ljung-Box result.

These summaries do not replace graphical residual inspection, checks for heteroskedasticity, structural instability, heavy tails, or model misspecification.

## Example

```python
from time_series_additive_models.diagnostics import summarize_residuals

diagnostics = summarize_residuals(
    [0.2, -0.1, 0.05, -0.2, 0.1, -0.05],
    lags=3,
)

print(diagnostics.autocorrelation)
print(diagnostics.ljung_box.statistic)
print(diagnostics.ljung_box.p_value)
```

The Ljung-Box p-value is a diagnostic aid, not a binary certification that residuals are independent.
