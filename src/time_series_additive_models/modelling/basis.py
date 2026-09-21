"""Basis construction for transparent additive time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_1d_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Convert array-like input to a finite, non-empty one-dimensional float array."""
    try:
        array: FloatArray = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _validate_nonnegative_int(value: int, *, name: str) -> None:
    """Validate an integer parameter that may be zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_int(value: int, *, name: str) -> None:
    """Validate a strictly positive integer parameter."""
    _validate_nonnegative_int(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True, slots=True)
class TimeTransform:
    """Affine transformation used to scale a time coordinate.

    The default fitted transform maps the training interval to [0, 1].
    Reusing the same transform for future observations ensures that training
    and prediction use an identical time scale.
    """

    origin: float
    scale: float

    def __post_init__(self) -> None:
        """Validate transformation parameters."""
        if not isfinite(self.origin):
            raise ValueError("origin must be finite.")
        if not isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and strictly positive.")

    @classmethod
    def fit(cls, time: ArrayLike) -> TimeTransform:
        """Fit a transform that maps the observed time span to [0, 1]."""
        values = _as_1d_float_array(time, name="time")
        origin = float(np.min(values))
        scale = float(np.max(values) - origin)

        if scale <= 0.0:
            raise ValueError("time must contain at least two distinct values.")

        return cls(origin=origin, scale=scale)

    def transform(self, time: ArrayLike) -> FloatArray:
        """Transform time values using the fitted affine mapping."""
        values = _as_1d_float_array(time, name="time")
        return np.asarray((values - self.origin) / self.scale, dtype=np.float64)

    def inverse_transform(self, scaled_time: ArrayLike) -> FloatArray:
        """Map scaled time values back to the original coordinate."""
        values = _as_1d_float_array(scaled_time, name="scaled_time")
        return np.asarray(self.origin + self.scale * values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BasisMatrix:
    """Named two-dimensional design matrix."""

    values: FloatArray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze matrix data."""
        array: FloatArray = np.asarray(self.values, dtype=np.float64)

        if array.ndim != 2:
            raise ValueError("values must be a two-dimensional array.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("values must have at least one row and one column.")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite values.")
        if len(self.columns) != array.shape[1]:
            raise ValueError("columns must match the number of matrix columns.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique.")

        frozen = np.array(array, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)


def polynomial_trend_basis(
    time: ArrayLike,
    degree: int,
    *,
    transform: TimeTransform | None = None,
    include_intercept: bool = True,
) -> BasisMatrix:
    """Construct a scaled polynomial basis for smooth deterministic trend.

    Parameters
    ----------
    time:
        One-dimensional numeric time coordinate.
    degree:
        Highest polynomial degree.
    transform:
        Optional pre-fitted time transform. When omitted, a transform is fitted
        to time. Supply the training transform again when creating a basis for
        future observations.
    include_intercept:
        Whether to include the constant column.
    """
    _validate_nonnegative_int(degree, name="degree")
    if degree == 0 and not include_intercept:
        raise ValueError("degree=0 with include_intercept=False creates no columns.")
    if transform is not None and not isinstance(transform, TimeTransform):
        raise TypeError("transform must be a TimeTransform or None.")

    fitted_transform = transform if transform is not None else TimeTransform.fit(time)
    scaled = fitted_transform.transform(time)
    matrix = np.asarray(
        np.vander(scaled, N=degree + 1, increasing=True),
        dtype=np.float64,
    )

    start = 0 if include_intercept else 1
    selected = matrix[:, start:]
    names = ["intercept"] if include_intercept else []
    names.extend(f"trend_power_{power}" for power in range(1, degree + 1))

    return BasisMatrix(values=selected, columns=tuple(names))


def piecewise_linear_trend_basis(
    time: ArrayLike,
    knots: ArrayLike,
    *,
    transform: TimeTransform | None = None,
    include_intercept: bool = True,
    include_linear: bool = True,
) -> BasisMatrix:
    """Construct a continuous piecewise-linear trend basis.

    Each knot contributes a hinge function max(0, t - knot) on the scaled
    time coordinate. Knots are specified in the original time units.
    """
    if transform is not None and not isinstance(transform, TimeTransform):
        raise TypeError("transform must be a TimeTransform or None.")

    fitted_transform = transform if transform is not None else TimeTransform.fit(time)
    scaled_time = fitted_transform.transform(time)
    knot_values = _as_1d_float_array(knots, name="knots")

    if np.any(np.diff(knot_values) <= 0.0):
        raise ValueError("knots must be strictly increasing.")

    lower = fitted_transform.origin
    upper = fitted_transform.origin + fitted_transform.scale
    if np.any((knot_values <= lower) | (knot_values >= upper)):
        raise ValueError("knots must lie strictly inside the fitted time interval.")

    scaled_knots = fitted_transform.transform(knot_values)

    components: list[FloatArray] = []
    names: list[str] = []

    if include_intercept:
        components.append(np.ones_like(scaled_time, dtype=np.float64))
        names.append("intercept")
    if include_linear:
        components.append(scaled_time)
        names.append("trend_linear")

    for index, knot in enumerate(scaled_knots, start=1):
        components.append(np.maximum(0.0, scaled_time - knot))
        names.append(f"trend_hinge_{index}")

    matrix = np.asarray(np.column_stack(components), dtype=np.float64)
    return BasisMatrix(values=matrix, columns=tuple(names))


def fourier_seasonal_basis(
    time: ArrayLike,
    *,
    period: float,
    order: int,
    origin: float = 0.0,
) -> BasisMatrix:
    """Construct Fourier sine/cosine pairs for periodic structure.

    period and time must use the same units. The function makes no assumption
    about regular spacing, so the caller is responsible for choosing a period
    meaningful for the supplied time coordinate.
    """
    values = _as_1d_float_array(time, name="time")
    _validate_positive_int(order, name="order")

    if not isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and strictly positive.")
    if not isfinite(origin):
        raise ValueError("origin must be finite.")

    components: list[FloatArray] = []
    names: list[str] = []

    for harmonic in range(1, order + 1):
        angle = 2.0 * pi * harmonic * (values - origin) / period
        components.append(np.sin(angle))
        components.append(np.cos(angle))
        names.extend((f"seasonal_sin_{harmonic}", f"seasonal_cos_{harmonic}"))

    matrix = np.asarray(np.column_stack(components), dtype=np.float64)
    return BasisMatrix(values=matrix, columns=tuple(names))


def combine_basis_matrices(*bases: BasisMatrix) -> BasisMatrix:
    """Combine compatible basis matrices column-wise."""
    if not bases:
        raise ValueError("at least one basis matrix is required.")
    if not all(isinstance(basis, BasisMatrix) for basis in bases):
        raise TypeError("all arguments must be BasisMatrix instances.")

    row_count = bases[0].values.shape[0]
    if any(basis.values.shape[0] != row_count for basis in bases[1:]):
        raise ValueError("all basis matrices must have the same number of rows.")

    columns = tuple(column for basis in bases for column in basis.columns)
    if len(set(columns)) != len(columns):
        raise ValueError("combined basis column names must be unique.")

    matrix = np.asarray(
        np.column_stack([basis.values for basis in bases]),
        dtype=np.float64,
    )
    return BasisMatrix(values=matrix, columns=columns)
