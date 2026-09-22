"""Ordinary least-squares estimation for named additive design matrices."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from .basis import BasisMatrix, FloatArray


class RankDeficientDesignError(ValueError):
    """Raised when a design matrix does not have full column rank."""


def _as_response(values: ArrayLike) -> FloatArray:
    """Convert a response to a finite, non-empty one-dimensional float array."""
    try:
        array: FloatArray = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("response must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError("response must be one-dimensional.")
    if array.size == 0:
        raise ValueError("response must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("response must contain only finite values.")

    return array


def _freeze(array: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy of an array-like value."""
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class OLSResult:
    """Fitted OLS model with the classical homoskedastic covariance estimate."""

    coefficients: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    fitted_values: FloatArray
    residuals: FloatArray
    columns: tuple[str, ...]
    rank: int
    dof_resid: int
    residual_variance: float
    rss: float
    condition_number: float

    def __post_init__(self) -> None:
        """Freeze arrays stored in the fitted result."""
        object.__setattr__(self, "coefficients", _freeze(self.coefficients))
        object.__setattr__(self, "covariance", _freeze(self.covariance))
        object.__setattr__(self, "standard_errors", _freeze(self.standard_errors))
        object.__setattr__(self, "fitted_values", _freeze(self.fitted_values))
        object.__setattr__(self, "residuals", _freeze(self.residuals))

    @property
    def nobs(self) -> int:
        """Number of observations used for fitting."""
        return int(self.fitted_values.size)

    @property
    def nparams(self) -> int:
        """Number of fitted coefficients."""
        return int(self.coefficients.size)

    def predict(self, design: BasisMatrix) -> FloatArray:
        """Predict from a design matrix with matching named columns."""
        if not isinstance(design, BasisMatrix):
            raise TypeError("design must be a BasisMatrix.")
        if design.columns != self.columns:
            raise ValueError("prediction design columns must exactly match fitted columns.")

        predictions = np.asarray(design.values @ self.coefficients, dtype=np.float64)
        return _freeze(predictions)


def fit_ols(
    design: BasisMatrix,
    response: ArrayLike,
    *,
    rcond: float | None = None,
) -> OLSResult:
    """Fit ordinary least squares to an explicit named design matrix.

    The supplied design must already contain every desired component, including
    an intercept when one is required. Classical standard errors use the
    homoskedastic covariance estimate sigma^2 times the inverse of X-transpose X.
    """
    if not isinstance(design, BasisMatrix):
        raise TypeError("design must be a BasisMatrix.")

    rcond_value: float | None = None
    if rcond is not None:
        if isinstance(rcond, bool) or not isinstance(rcond, Real):
            raise TypeError("rcond must be a non-negative finite float or None.")
        rcond_value = float(rcond)
        if not isfinite(rcond_value) or rcond_value < 0.0:
            raise ValueError("rcond must be non-negative and finite.")

    y = _as_response(response)
    x = design.values
    nobs, nparams = x.shape

    if y.size != nobs:
        raise ValueError("response length must match the number of design rows.")
    if nobs <= nparams:
        raise ValueError("ordinary least squares requires positive residual degrees of freedom.")

    coefficients_raw, _, rank_raw, _ = np.linalg.lstsq(x, y, rcond=rcond_value)
    rank = int(rank_raw)
    if rank < nparams:
        raise RankDeficientDesignError("design matrix must have full column rank.")

    coefficients = np.asarray(coefficients_raw, dtype=np.float64)
    fitted = np.asarray(x @ coefficients, dtype=np.float64)
    residuals = np.asarray(y - fitted, dtype=np.float64)
    rss = float(residuals @ residuals)
    dof_resid = nobs - nparams
    residual_variance = rss / dof_resid

    _, upper = np.linalg.qr(x, mode="reduced")
    inverse_upper = np.linalg.inv(upper)
    xtx_inverse = inverse_upper @ inverse_upper.T
    covariance = np.asarray(residual_variance * xtx_inverse, dtype=np.float64)
    standard_errors = np.asarray(np.sqrt(np.diag(covariance)), dtype=np.float64)
    condition_number = float(np.linalg.cond(x))

    return OLSResult(
        coefficients=coefficients,
        covariance=covariance,
        standard_errors=standard_errors,
        fitted_values=fitted,
        residuals=residuals,
        columns=design.columns,
        rank=rank,
        dof_resid=dof_resid,
        residual_variance=residual_variance,
        rss=rss,
        condition_number=condition_number,
    )
