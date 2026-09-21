"""Smoke tests for the installed package."""

from __future__ import annotations

import time_series_additive_models


def test_package_exposes_version() -> None:
    """The installed package should expose its public version."""
    assert time_series_additive_models.__version__ == "0.1.0"


def test_package_public_api_declares_version() -> None:
    """Keep the initial public API explicit and minimal."""
    assert time_series_additive_models.__all__ == ["__version__"]
