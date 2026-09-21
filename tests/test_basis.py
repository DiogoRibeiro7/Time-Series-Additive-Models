"""Tests for additive basis construction."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from time_series_additive_models.modelling.basis import (
    BasisMatrix,
    TimeTransform,
    combine_basis_matrices,
    fourier_seasonal_basis,
    piecewise_linear_trend_basis,
    polynomial_trend_basis,
)


def test_time_transform_maps_training_span_to_unit_interval() -> None:
    transform = TimeTransform.fit([10.0, 15.0, 20.0])
    transformed = transform.transform([10.0, 15.0, 20.0])
    np.testing.assert_allclose(transformed, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        transform.inverse_transform(transformed),
        [10.0, 15.0, 20.0],
    )


def test_time_transform_can_be_reused_for_future_observations() -> None:
    transform = TimeTransform.fit([0.0, 10.0])
    np.testing.assert_allclose(transform.transform([15.0, 20.0]), [1.5, 2.0])


@pytest.mark.parametrize(
    "time",
    [[], [1.0], [1.0, 1.0], [0.0, np.nan], [[0.0, 1.0]]],
)
def test_time_transform_rejects_invalid_training_coordinates(time: object) -> None:
    with pytest.raises(ValueError):
        TimeTransform.fit(time)


def test_time_transform_rejects_non_numeric_input() -> None:
    with pytest.raises(TypeError, match="numeric"):
        TimeTransform.fit(  # type: ignore[arg-type]
            ["not-a-number", "still-not-a-number"]
        )


@pytest.mark.parametrize(
    ("origin", "scale"),
    [(np.nan, 1.0), (0.0, np.nan), (0.0, 0.0), (0.0, -1.0)],
)
def test_time_transform_validates_parameters(origin: float, scale: float) -> None:
    with pytest.raises(ValueError):
        TimeTransform(origin=origin, scale=scale)


def test_polynomial_trend_basis_has_expected_values() -> None:
    transform = TimeTransform.fit([0.0, 2.0])
    basis = polynomial_trend_basis([0.0, 1.0, 2.0], degree=2, transform=transform)

    assert basis.columns == ("intercept", "trend_power_1", "trend_power_2")
    np.testing.assert_allclose(
        basis.values,
        [[1.0, 0.0, 0.0], [1.0, 0.5, 0.25], [1.0, 1.0, 1.0]],
    )
    assert not basis.values.flags.writeable


def test_polynomial_trend_basis_can_drop_intercept() -> None:
    basis = polynomial_trend_basis([0.0, 1.0, 2.0], degree=1, include_intercept=False)
    assert basis.columns == ("trend_power_1",)
    assert basis.values.shape == (3, 1)


@pytest.mark.parametrize("degree", [-1, True])
def test_polynomial_trend_basis_rejects_invalid_degree(degree: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        polynomial_trend_basis(  # type: ignore[arg-type]
            [0.0, 1.0],
            degree=degree,
        )


def test_polynomial_trend_basis_rejects_empty_design() -> None:
    with pytest.raises(ValueError, match="creates no columns"):
        polynomial_trend_basis([0.0, 1.0], degree=0, include_intercept=False)


def test_polynomial_trend_basis_rejects_invalid_transform() -> None:
    with pytest.raises(TypeError, match="TimeTransform"):
        polynomial_trend_basis(
            [0.0, 1.0],
            degree=1,
            transform=object(),  # type: ignore[arg-type]
        )


def test_piecewise_linear_trend_basis_builds_hinges() -> None:
    transform = TimeTransform.fit([0.0, 4.0])
    basis = piecewise_linear_trend_basis(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        knots=[2.0],
        transform=transform,
    )

    assert basis.columns == ("intercept", "trend_linear", "trend_hinge_1")
    np.testing.assert_allclose(
        basis.values,
        [
            [1.0, 0.00, 0.00],
            [1.0, 0.25, 0.00],
            [1.0, 0.50, 0.00],
            [1.0, 0.75, 0.25],
            [1.0, 1.00, 0.50],
        ],
    )


def test_piecewise_linear_trend_basis_supports_hinge_only_design() -> None:
    basis = piecewise_linear_trend_basis(
        [0.0, 1.0, 2.0],
        knots=[1.0],
        include_intercept=False,
        include_linear=False,
    )
    assert basis.columns == ("trend_hinge_1",)
    np.testing.assert_allclose(basis.values[:, 0], [0.0, 0.0, 0.5])


@pytest.mark.parametrize(
    "knots",
    [[2.0, 1.0], [1.0, 1.0], [0.0, 2.0], [2.0, 4.0]],
)
def test_piecewise_linear_trend_basis_rejects_invalid_knots(knots: list[float]) -> None:
    transform = TimeTransform.fit([0.0, 4.0])
    with pytest.raises(ValueError):
        piecewise_linear_trend_basis(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            knots=knots,
            transform=transform,
        )


def test_piecewise_linear_trend_basis_rejects_invalid_transform() -> None:
    with pytest.raises(TypeError, match="TimeTransform"):
        piecewise_linear_trend_basis(
            [0.0, 1.0, 2.0],
            knots=[1.0],
            transform=object(),  # type: ignore[arg-type]
        )


def test_fourier_seasonal_basis_matches_quarter_cycle() -> None:
    basis = fourier_seasonal_basis([0.0, 1.0, 2.0, 3.0], period=4.0, order=1)
    assert basis.columns == ("seasonal_sin_1", "seasonal_cos_1")
    np.testing.assert_allclose(
        basis.values,
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        atol=1e-12,
    )


def test_fourier_seasonal_basis_supports_multiple_harmonics() -> None:
    basis = fourier_seasonal_basis([0.0, 1.0], period=12.0, order=3)
    assert basis.values.shape == (2, 6)
    assert basis.columns[-2:] == ("seasonal_sin_3", "seasonal_cos_3")


@pytest.mark.parametrize(
    ("period", "order", "origin"),
    [(0.0, 1, 0.0), (-1.0, 1, 0.0), (12.0, 0, 0.0), (12.0, 1, np.nan)],
)
def test_fourier_seasonal_basis_rejects_invalid_parameters(
    period: float,
    order: int,
    origin: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        fourier_seasonal_basis(
            [0.0, 1.0],
            period=period,
            order=order,
            origin=origin,
        )


@pytest.mark.parametrize(
    ("values", "columns"),
    [
        (np.ones(2), ("x",)),
        (np.empty((0, 1)), ("x",)),
        (np.array([[np.nan]]), ("x",)),
        (np.ones((2, 2)), ("x",)),
        (np.ones((2, 2)), ("x", "x")),
    ],
)
def test_basis_matrix_rejects_invalid_matrix_metadata(
    values: NDArray[np.float64],
    columns: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        BasisMatrix(values=values, columns=columns)


def test_combine_basis_matrices_stacks_components() -> None:
    trend = polynomial_trend_basis([0.0, 1.0, 2.0], degree=1)
    seasonal = fourier_seasonal_basis([0.0, 1.0, 2.0], period=4.0, order=1)
    combined = combine_basis_matrices(trend, seasonal)

    assert combined.values.shape == (3, 4)
    assert combined.columns == (
        "intercept",
        "trend_power_1",
        "seasonal_sin_1",
        "seasonal_cos_1",
    )


def test_combine_basis_matrices_rejects_no_arguments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        combine_basis_matrices()


def test_combine_basis_matrices_rejects_non_basis_argument() -> None:
    basis = BasisMatrix(np.ones((2, 1)), ("x",))
    with pytest.raises(TypeError, match="BasisMatrix"):
        combine_basis_matrices(basis, object())  # type: ignore[arg-type]


def test_combine_basis_matrices_rejects_incompatible_rows() -> None:
    left = BasisMatrix(np.ones((2, 1)), ("left",))
    right = BasisMatrix(np.ones((3, 1)), ("right",))
    with pytest.raises(ValueError, match="same number of rows"):
        combine_basis_matrices(left, right)


def test_combine_basis_matrices_rejects_duplicate_column_names() -> None:
    left = BasisMatrix(np.ones((2, 1)), ("duplicate",))
    right = BasisMatrix(np.ones((2, 1)), ("duplicate",))
    with pytest.raises(ValueError, match="unique"):
        combine_basis_matrices(left, right)
