"""Tests for ordinary least-squares estimation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike

from time_series_additive_models.modelling.basis import BasisMatrix
from time_series_additive_models.modelling.estimation import (
    RankDeficientDesignError,
    fit_ols,
)


def _simple_design() -> BasisMatrix:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return BasisMatrix(
        values=np.column_stack([np.ones_like(x), x]),
        columns=("intercept", "trend_linear"),
    )


def test_fit_ols_recovers_known_coefficients_and_classical_covariance() -> None:
    design = _simple_design()
    noise = np.array([1.0, -2.0, 2.0, -2.0, 1.0])
    response = 2.0 + 3.0 * design.values[:, 1] + noise

    result = fit_ols(design, response)

    np.testing.assert_allclose(result.coefficients, [2.0, 3.0])
    assert result.columns == ("intercept", "trend_linear")
    assert result.rank == 2
    assert result.nobs == 5
    assert result.nparams == 2
    assert result.dof_resid == 3
    assert result.rss == pytest.approx(14.0)
    assert result.residual_variance == pytest.approx(14.0 / 3.0)
    np.testing.assert_allclose(
        result.covariance,
        [[14.0 / 15.0, 0.0], [0.0, 7.0 / 15.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.standard_errors,
        np.sqrt([14.0 / 15.0, 7.0 / 15.0]),
    )
    np.testing.assert_allclose(result.residuals, noise)
    assert result.condition_number >= 1.0


def test_fit_ols_result_arrays_are_immutable() -> None:
    result = fit_ols(_simple_design(), [-3.0, -1.0, 1.0, 3.0, 5.0])

    assert not result.coefficients.flags.writeable
    assert not result.covariance.flags.writeable
    assert not result.standard_errors.flags.writeable
    assert not result.fitted_values.flags.writeable
    assert not result.residuals.flags.writeable


def test_predict_requires_matching_named_columns() -> None:
    result = fit_ols(_simple_design(), [-3.0, -1.0, 1.0, 3.0, 5.0])
    future = BasisMatrix(
        values=np.array([[1.0, 3.0], [1.0, 4.0]]),
        columns=("intercept", "trend_linear"),
    )

    predictions = result.predict(future)

    np.testing.assert_allclose(predictions, [7.0, 9.0])
    assert not predictions.flags.writeable


def test_predict_rejects_non_basis_design() -> None:
    result = fit_ols(_simple_design(), [-3.0, -1.0, 1.0, 3.0, 5.0])

    with pytest.raises(TypeError, match="BasisMatrix"):
        result.predict(cast(BasisMatrix, object()))


def test_predict_rejects_mismatched_columns() -> None:
    result = fit_ols(_simple_design(), [-3.0, -1.0, 1.0, 3.0, 5.0])
    mismatched = BasisMatrix(
        values=np.array([[1.0, 3.0]]),
        columns=("intercept", "other"),
    )

    with pytest.raises(ValueError, match="exactly match"):
        result.predict(mismatched)


def test_fit_ols_rejects_non_basis_design() -> None:
    with pytest.raises(TypeError, match="BasisMatrix"):
        fit_ols(cast(BasisMatrix, object()), [1.0, 2.0, 3.0])


def test_fit_ols_rejects_rank_deficient_design() -> None:
    design = BasisMatrix(
        values=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        columns=("a", "b"),
    )

    with pytest.raises(RankDeficientDesignError):
        fit_ols(design, [1.0, 2.0, 3.0])


def test_fit_ols_rejects_nonpositive_residual_degrees_of_freedom() -> None:
    design = BasisMatrix(values=np.eye(2), columns=("a", "b"))

    with pytest.raises(ValueError, match="residual degrees"):
        fit_ols(design, [1.0, 2.0])


@pytest.mark.parametrize(
    "response",
    [[], [1.0, np.nan, 3.0, 4.0, 5.0], [[1.0], [2.0], [3.0], [4.0], [5.0]]],
)
def test_fit_ols_rejects_invalid_response(response: ArrayLike) -> None:
    with pytest.raises(ValueError):
        fit_ols(_simple_design(), response)


def test_fit_ols_rejects_non_numeric_response() -> None:
    with pytest.raises(TypeError, match="numeric"):
        fit_ols(_simple_design(), ["a", "b", "c", "d", "e"])


def test_fit_ols_rejects_response_length_mismatch() -> None:
    with pytest.raises(ValueError, match="response length"):
        fit_ols(_simple_design(), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("rcond", [-1.0, np.nan])
def test_fit_ols_rejects_invalid_numeric_rcond(rcond: float) -> None:
    with pytest.raises(ValueError):
        fit_ols(_simple_design(), [-3.0, -1.0, 1.0, 3.0, 5.0], rcond=rcond)


def test_fit_ols_rejects_non_numeric_rcond() -> None:
    with pytest.raises(TypeError):
        fit_ols(
            _simple_design(),
            [-3.0, -1.0, 1.0, 3.0, 5.0],
            rcond=cast(float, "invalid"),
        )


def test_fit_ols_accepts_explicit_rcond() -> None:
    result = fit_ols(
        _simple_design(),
        [-3.0, -1.0, 1.0, 3.0, 5.0],
        rcond=1e-12,
    )

    np.testing.assert_allclose(result.coefficients, [1.0, 2.0])
