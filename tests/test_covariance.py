"""Tests for robust OLS covariance estimators."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from time_series_additive_models.modelling import BasisMatrix, OLSResult, fit_ols
from time_series_additive_models.modelling.covariance import (
    CovarianceResult,
    HCType,
    heteroskedasticity_consistent_covariance,
    newey_west_covariance,
)


def _fit() -> tuple[BasisMatrix, OLSResult]:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    design = BasisMatrix(
        values=np.column_stack([np.ones_like(x), x]),
        columns=("intercept", "trend_linear"),
    )
    noise = np.array([1.0, -2.0, 2.0, -2.0, 1.0])
    response = 2.0 + 3.0 * x + noise
    return design, fit_ols(design, response)


def _manual_hc(
    design: BasisMatrix,
    residuals: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    x = design.values
    bread = np.linalg.inv(x.T @ x)
    meat = x.T @ (x * weights[:, np.newaxis])
    return np.asarray(bread @ meat @ bread, dtype=np.float64)


def test_hc0_matches_direct_sandwich_formula() -> None:
    design, result = _fit()
    robust = heteroskedasticity_consistent_covariance(result, design, kind="HC0")
    expected = _manual_hc(design, result.residuals, result.residuals**2)
    np.testing.assert_allclose(robust.covariance, expected)
    np.testing.assert_allclose(robust.standard_errors, np.sqrt(np.diag(expected)))
    assert robust.method == "HC0"
    assert robust.small_sample is False


def test_hc1_is_hc0_with_degrees_of_freedom_scaling() -> None:
    design, result = _fit()
    hc0 = heteroskedasticity_consistent_covariance(result, design, kind="HC0")
    hc1 = heteroskedasticity_consistent_covariance(result, design, kind="HC1")
    np.testing.assert_allclose(
        hc1.covariance,
        hc0.covariance * result.nobs / result.dof_resid,
    )
    assert hc1.small_sample is True


@pytest.mark.parametrize(("kind", "power"), [("HC2", 1), ("HC3", 2)])
def test_hc2_hc3_match_leverage_adjustment(kind: HCType, power: int) -> None:
    design, result = _fit()
    x = design.values
    bread = np.linalg.inv(x.T @ x)
    leverage = np.sum((x @ bread) * x, axis=1)
    weights = result.residuals**2 / (1.0 - leverage) ** power
    expected = _manual_hc(design, result.residuals, weights)
    robust = heteroskedasticity_consistent_covariance(result, design, kind=kind)
    np.testing.assert_allclose(robust.covariance, expected)


def test_covariance_result_arrays_are_immutable() -> None:
    design, result = _fit()
    robust = heteroskedasticity_consistent_covariance(result, design)
    assert not robust.covariance.flags.writeable
    assert not robust.standard_errors.flags.writeable


def test_hc_rejects_invalid_kind() -> None:
    design, result = _fit()
    with pytest.raises(ValueError, match="kind"):
        heteroskedasticity_consistent_covariance(
            result,
            design,
            kind=cast(HCType, "HC9"),
        )


def test_hc2_rejects_unit_leverage() -> None:
    x = np.array([0.0, 0.0, 1.0])
    design = BasisMatrix(
        values=np.column_stack([np.ones_like(x), x]),
        columns=("intercept", "indicator"),
    )
    result = fit_ols(design, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="leverage"):
        heteroskedasticity_consistent_covariance(result, design, kind="HC2")


def test_newey_west_matches_direct_bartlett_formula() -> None:
    design, result = _fit()
    x = design.values
    scores = x * result.residuals[:, np.newaxis]
    meat = scores.T @ scores
    max_lag = 2
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    expected = bread @ meat @ bread

    robust = newey_west_covariance(
        result,
        design,
        max_lag=max_lag,
        small_sample=False,
    )

    np.testing.assert_allclose(robust.covariance, expected)
    assert robust.method == "Newey-West"
    assert robust.max_lag == 2
    assert robust.small_sample is False


def test_newey_west_small_sample_scales_covariance() -> None:
    design, result = _fit()
    raw = newey_west_covariance(result, design, max_lag=1, small_sample=False)
    corrected = newey_west_covariance(result, design, max_lag=1, small_sample=True)
    np.testing.assert_allclose(
        corrected.covariance,
        raw.covariance * result.nobs / result.dof_resid,
    )


@pytest.mark.parametrize("max_lag", [-1, 5])
def test_newey_west_rejects_invalid_lag_range(max_lag: int) -> None:
    design, result = _fit()
    with pytest.raises(ValueError):
        newey_west_covariance(result, design, max_lag=max_lag)


def test_newey_west_rejects_non_integer_lag() -> None:
    design, result = _fit()
    with pytest.raises(TypeError):
        newey_west_covariance(result, design, max_lag=cast(int, 1.5))


def test_newey_west_rejects_non_boolean_small_sample() -> None:
    design, result = _fit()
    with pytest.raises(TypeError):
        newey_west_covariance(
            result,
            design,
            max_lag=1,
            small_sample=cast(bool, 1),
        )


def test_covariance_rejects_mismatched_columns() -> None:
    design, result = _fit()
    wrong = BasisMatrix(design.values, ("intercept", "wrong"))
    with pytest.raises(ValueError, match="columns"):
        heteroskedasticity_consistent_covariance(result, wrong)


def test_covariance_rejects_mismatched_rows() -> None:
    design, result = _fit()
    short = BasisMatrix(design.values[:-1], design.columns)
    with pytest.raises(ValueError, match="rows"):
        heteroskedasticity_consistent_covariance(result, short)


def test_covariance_rejects_rank_deficient_design() -> None:
    _, result = _fit()
    rank_deficient = BasisMatrix(
        np.ones((result.nobs, result.nparams)),
        result.columns,
    )
    with pytest.raises(ValueError, match="full column rank"):
        heteroskedasticity_consistent_covariance(result, rank_deficient)


def test_covariance_result_validates_shapes() -> None:
    with pytest.raises(ValueError):
        CovarianceResult(
            covariance=np.ones((2, 3)),
            standard_errors=np.ones(2),
            method="invalid",
        )

