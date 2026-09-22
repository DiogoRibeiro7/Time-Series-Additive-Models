"""Statistical modelling components."""

from .basis import (
    BasisMatrix,
    TimeTransform,
    combine_basis_matrices,
    fourier_seasonal_basis,
    piecewise_linear_trend_basis,
    polynomial_trend_basis,
)
from .covariance import (
    CovarianceResult,
    HCType,
    heteroskedasticity_consistent_covariance,
    newey_west_covariance,
)
from .estimation import OLSResult, RankDeficientDesignError, fit_ols

__all__ = [
    "BasisMatrix",
    "CovarianceResult",
    "HCType",
    "OLSResult",
    "RankDeficientDesignError",
    "TimeTransform",
    "combine_basis_matrices",
    "fit_ols",
    "fourier_seasonal_basis",
    "heteroskedasticity_consistent_covariance",
    "newey_west_covariance",
    "piecewise_linear_trend_basis",
    "polynomial_trend_basis",
]
