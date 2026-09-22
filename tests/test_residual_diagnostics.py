"""Tests for residual diagnostics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike

from time_series_additive_models.diagnostics import (
    autocorrelation,
    ljung_box,
    summarize_residuals,
)


def test_autocorrelation_matches_manual_calculation() -> None:
    residuals = np.array([1.0, -1.0, 1.0, -1.0])
    acf = autocorrelation(residuals, lags=2)
    np.testing.assert_allclose(acf, [1.0, -0.75, 0.5])


def test_autocorrelation_can_skip_demeaning() -> None:
    residuals = np.array([1.0, 2.0, 3.0, 4.0])
    acf = autocorrelation(residuals, lags=1, demean=False)
    expected = (2.0 * 1.0 + 3.0 * 2.0 + 4.0 * 3.0) / 30.0
    assert acf[0] == 1.0
    assert acf[1] == pytest.approx(expected)
    assert not acf.flags.writeable


def test_autocorrelation_rejects_zero_variance_sequence() -> None:
    with pytest.raises(ValueError, match="zero-variance"):
        autocorrelation([2.0, 2.0, 2.0], lags=1)


@pytest.mark.parametrize(
    "residuals",
    [[], [1.0], [1.0, np.nan], [[1.0, 2.0], [3.0, 4.0]]],
)
def test_diagnostics_reject_invalid_residuals(residuals: ArrayLike) -> None:
    with pytest.raises(ValueError):
        autocorrelation(residuals, lags=1)


def test_diagnostics_reject_non_numeric_residuals() -> None:
    with pytest.raises(TypeError, match="numeric"):
        autocorrelation(["a", "b"], lags=1)


@pytest.mark.parametrize("lags", [0, -1, 4])
def test_autocorrelation_rejects_invalid_lag_range(lags: int) -> None:
    with pytest.raises(ValueError):
        autocorrelation([1.0, 2.0, 3.0, 4.0], lags=lags)


def test_autocorrelation_rejects_non_integer_lags() -> None:
    with pytest.raises(TypeError):
        autocorrelation([1.0, 2.0, 3.0], lags=cast(int, 1.5))


def test_ljung_box_matches_direct_formula() -> None:
    residuals = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    acf = autocorrelation(residuals, lags=2)
    nobs = residuals.size
    expected = nobs * (nobs + 2) * sum(
        (acf[lag] ** 2) / (nobs - lag) for lag in range(1, 3)
    )
    result = ljung_box(residuals, lags=2)
    assert result.lag == 2
    assert result.degrees_of_freedom == 2
    assert result.statistic == pytest.approx(expected)
    assert 0.0 <= result.p_value <= 1.0


def test_ljung_box_applies_model_df_adjustment() -> None:
    result = ljung_box(
        [0.2, -0.1, 0.1, -0.2, 0.15, -0.05],
        lags=3,
        model_df=1,
    )
    assert result.degrees_of_freedom == 2


@pytest.mark.parametrize("model_df", [-1, 3])
def test_ljung_box_rejects_invalid_model_df(model_df: int) -> None:
    with pytest.raises(ValueError):
        ljung_box([1.0, -1.0, 1.0, -1.0], lags=3, model_df=model_df)


def test_ljung_box_rejects_non_integer_model_df() -> None:
    with pytest.raises(TypeError):
        ljung_box(
            [1.0, -1.0, 1.0, -1.0],
            lags=2,
            model_df=cast(int, 1.5),
        )


def test_summarize_residuals_returns_scale_and_serial_diagnostics() -> None:
    residuals = np.array([1.0, -2.0, 2.0, -1.0])
    result = summarize_residuals(residuals, lags=2)
    assert result.nobs == 4
    assert result.mean == pytest.approx(0.0)
    assert result.standard_deviation == pytest.approx(np.std(residuals, ddof=1))
    assert result.rmse == pytest.approx(np.sqrt(np.mean(residuals**2)))
    assert result.mae == pytest.approx(1.5)
    assert result.autocorrelation.shape == (3,)
    assert not result.autocorrelation.flags.writeable
    assert result.ljung_box.lag == 2
