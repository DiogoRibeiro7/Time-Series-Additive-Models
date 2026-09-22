"""Robust covariance estimators for ordinary least-squares fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .basis import BasisMatrix, FloatArray
from .estimation import OLSResult, RankDeficientDesignError

HCType = Literal["HC0", "HC1", "HC2", "HC3"]


def _freeze(array: FloatArray) -> FloatArray:
    """Return an immutable float64 copy."""
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class CovarianceResult:
    """Alternative covariance estimate for an existing OLS coefficient vector."""

    covariance: FloatArray
    standard_errors: FloatArray
    method: str
    max_lag: int | None = None
    small_sample: bool = False

    def __post_init__(self) -> None:
        """Freeze covariance outputs."""
        covariance = np.asarray(self.covariance, dtype=np.float64)
        standard_errors = np.asarray(self.standard_errors, dtype=np.float64)

        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix.")
        if standard_errors.ndim != 1 or standard_errors.size != covariance.shape[0]:
            raise ValueError("standard_errors must match covariance dimensions.")
        if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(standard_errors)):
            raise ValueError("covariance results must be finite.")

        object.__setattr__(self, "covariance", _freeze(covariance))
        object.__setattr__(self, "standard_errors", _freeze(standard_errors))


def _validate_fit_design(result: OLSResult, design: BasisMatrix) -> FloatArray:
    """Validate that a design matrix corresponds to an OLS fit."""
    if not isinstance(result, OLSResult):
        raise TypeError("result must be an OLSResult.")
    if not isinstance(design, BasisMatrix):
        raise TypeError("design must be a BasisMatrix.")
    if design.columns != result.columns:
        raise ValueError("design columns must exactly match fitted columns.")
    if design.values.shape[0] != result.nobs:
        raise ValueError("design rows must match the fitted number of observations.")
    if design.values.shape[1] != result.nparams:
        raise ValueError("design columns must match the fitted number of parameters.")
    if np.linalg.matrix_rank(design.values) < result.nparams:
        raise RankDeficientDesignError("design matrix must have full column rank.")

    return design.values


def _bread(design: FloatArray) -> FloatArray:
    """Compute the inverse cross-product matrix for a full-rank design."""
    _, upper = np.linalg.qr(design, mode="reduced")
    inverse_upper = np.linalg.inv(upper)
    return np.asarray(inverse_upper @ inverse_upper.T, dtype=np.float64)


def _covariance_result(
    covariance: FloatArray,
    *,
    method: str,
    max_lag: int | None = None,
    small_sample: bool = False,
) -> CovarianceResult:
    """Build a validated covariance result."""
    symmetric = np.asarray((covariance + covariance.T) / 2.0, dtype=np.float64)
    diagonal = np.diag(symmetric)
    if np.any(diagonal < -1e-12):
        raise ValueError("covariance matrix has a materially negative diagonal entry.")

    standard_errors = np.sqrt(np.maximum(diagonal, 0.0))
    return CovarianceResult(
        covariance=symmetric,
        standard_errors=np.asarray(standard_errors, dtype=np.float64),
        method=method,
        max_lag=max_lag,
        small_sample=small_sample,
    )


def heteroskedasticity_consistent_covariance(
    result: OLSResult,
    design: BasisMatrix,
    *,
    kind: HCType = "HC1",
) -> CovarianceResult:
    """Compute HC0, HC1, HC2, or HC3 covariance for an OLS fit."""
    x = _validate_fit_design(result, design)
    if kind not in {"HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("kind must be one of HC0, HC1, HC2, or HC3.")

    residuals = result.residuals
    nobs = result.nobs
    bread = _bread(x)
    squared_residuals = residuals**2

    if kind == "HC0":
        weights = squared_residuals
    elif kind == "HC1":
        weights = squared_residuals * (nobs / result.dof_resid)
    else:
        leverage = np.sum((x @ bread) * x, axis=1)
        one_minus_leverage = 1.0 - leverage
        if np.any(one_minus_leverage <= np.finfo(np.float64).eps):
            raise ValueError("HC2/HC3 covariance is undefined when leverage is one.")
        power = 1 if kind == "HC2" else 2
        weights = squared_residuals / (one_minus_leverage**power)

    meat = x.T @ (x * weights[:, np.newaxis])
    covariance = np.asarray(bread @ meat @ bread, dtype=np.float64)

    return _covariance_result(
        covariance,
        method=kind,
        small_sample=(kind == "HC1"),
    )


def newey_west_covariance(
    result: OLSResult,
    design: BasisMatrix,
    *,
    max_lag: int,
    small_sample: bool = True,
) -> CovarianceResult:
    """Compute Newey-West HAC covariance with Bartlett kernel weights."""
    x = _validate_fit_design(result, design)

    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError("max_lag must be an integer.")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")
    if max_lag >= result.nobs:
        raise ValueError("max_lag must be smaller than the number of observations.")
    if not isinstance(small_sample, bool):
        raise TypeError("small_sample must be boolean.")

    scores = x * result.residuals[:, np.newaxis]
    meat = np.asarray(scores.T @ scores, dtype=np.float64)

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    if small_sample:
        meat *= result.nobs / result.dof_resid

    bread = _bread(x)
    covariance = np.asarray(bread @ meat @ bread, dtype=np.float64)

    return _covariance_result(
        covariance,
        method="Newey-West",
        max_lag=max_lag,
        small_sample=small_sample,
    )
