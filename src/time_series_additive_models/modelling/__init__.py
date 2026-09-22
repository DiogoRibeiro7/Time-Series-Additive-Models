"""Statistical modelling components."""

from .basis import (
    BasisMatrix,
    TimeTransform,
    combine_basis_matrices,
    fourier_seasonal_basis,
    piecewise_linear_trend_basis,
    polynomial_trend_basis,
)
from .estimation import OLSResult, RankDeficientDesignError, fit_ols

__all__ = [
    "BasisMatrix",
    "OLSResult",
    "RankDeficientDesignError",
    "TimeTransform",
    "combine_basis_matrices",
    "fit_ols",
    "fourier_seasonal_basis",
    "piecewise_linear_trend_basis",
    "polynomial_trend_basis",
]
