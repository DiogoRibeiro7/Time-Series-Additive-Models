"""Residual diagnostics for additive time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]


def _as_residuals(values: ArrayLike) -> FloatArray:
    """Convert residuals to a finite non-empty one-dimensional float array."""
    try:
        residuals: FloatArray = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("residuals must contain numeric values.") from exc

    if residuals.ndim != 1:
        raise ValueError("residuals must be one-dimensional.")
    if residuals.size < 2:
        raise ValueError("residuals must contain at least two observations.")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values.")

    return residuals


def _validate_lags(lags: int, *, nobs: int) -> None:
    """Validate a maximum autocorrelation lag."""
    if isinstance(lags, bool) or not isinstance(lags, int):
        raise TypeError("lags must be an integer.")
    if lags < 1:
        raise ValueError("lags must be at least one.")
    if lags >= nobs:
        raise ValueError("lags must be smaller than the number of observations.")


@dataclass(frozen=True, slots=True)
class LjungBoxResult:
    """Ljung-Box portmanteau statistic at a single maximum lag."""

    lag: int
    statistic: float
    p_value: float
    degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Summary diagnostics computed directly from a residual sequence."""

    nobs: int
    mean: float
    standard_deviation: float
    rmse: float
    mae: float
    autocorrelation: FloatArray
    ljung_box: LjungBoxResult

    def __post_init__(self) -> None:
        """Freeze the autocorrelation vector stored in the result."""
        frozen = np.array(self.autocorrelation, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "autocorrelation", frozen)


def autocorrelation(
    residuals: ArrayLike,
    *,
    lags: int,
    demean: bool = True,
) -> FloatArray:
    """Estimate residual autocorrelation from lag zero through the requested lag."""
    values = _as_residuals(residuals)
    _validate_lags(lags, nobs=values.size)

    centered = values - np.mean(values) if demean else values.copy()
    denominator = float(centered @ centered)

    if denominator <= 0.0:
        raise ValueError("autocorrelation is undefined for a zero-variance sequence.")

    acf = np.empty(lags + 1, dtype=np.float64)
    acf[0] = 1.0

    for lag in range(1, lags + 1):
        acf[lag] = float(centered[lag:] @ centered[:-lag]) / denominator

    acf.setflags(write=False)
    return acf


def ljung_box(
    residuals: ArrayLike,
    *,
    lags: int,
    model_df: int = 0,
) -> LjungBoxResult:
    """Compute the Ljung-Box statistic for residual serial correlation."""
    values = _as_residuals(residuals)
    _validate_lags(lags, nobs=values.size)

    if isinstance(model_df, bool) or not isinstance(model_df, int):
        raise TypeError("model_df must be an integer.")
    if model_df < 0:
        raise ValueError("model_df must be non-negative.")

    degrees_of_freedom = lags - model_df
    if degrees_of_freedom <= 0:
        raise ValueError("lags - model_df must be positive.")

    acf = autocorrelation(values, lags=lags)
    nobs = values.size
    statistic = 0.0

    for lag in range(1, lags + 1):
        statistic += (acf[lag] ** 2) / (nobs - lag)

    statistic *= nobs * (nobs + 2)
    p_value = float(chi2.sf(statistic, degrees_of_freedom))

    return LjungBoxResult(
        lag=lags,
        statistic=float(statistic),
        p_value=p_value,
        degrees_of_freedom=degrees_of_freedom,
    )


def summarize_residuals(
    residuals: ArrayLike,
    *,
    lags: int,
    model_df: int = 0,
) -> ResidualDiagnostics:
    """Compute scale summaries, autocorrelation, and Ljung-Box diagnostics."""
    values = _as_residuals(residuals)
    acf = autocorrelation(values, lags=lags)
    lb = ljung_box(values, lags=lags, model_df=model_df)

    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    rmse = sqrt(float(np.mean(values**2)))
    mae = float(np.mean(np.abs(values)))

    summaries = (mean, standard_deviation, rmse, mae)
    if not all(isfinite(value) for value in summaries):
        raise ValueError("residual diagnostics produced a non-finite summary.")

    return ResidualDiagnostics(
        nobs=int(values.size),
        mean=mean,
        standard_deviation=standard_deviation,
        rmse=rmse,
        mae=mae,
        autocorrelation=acf,
        ljung_box=lb,
    )
