"""Repository-level regression tests for historical research artefacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "legacy" / "notebooks" / "additive_models_for_prediction.ipynb"
)

_LITERAL_API_KEY_ASSIGNMENT = re.compile(
    r"""quandl\.ApiConfig\.api_key\s*=\s*['"][^'"]+['"]"""
)


def _load_notebook() -> dict[str, object]:
    """Load the historical notebook as a typed JSON object."""
    raw: object = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise TypeError("The notebook root must be a JSON object.")

    return cast(dict[str, object], raw)


def _code_source(notebook: dict[str, object]) -> str:
    """Return the concatenated source of all code cells in a notebook."""
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TypeError("The notebook must contain a list of cells.")

    chunks: list[str] = []

    for cell_object in cells:
        if not isinstance(cell_object, dict):
            continue

        cell = cast(dict[str, object], cell_object)
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source")
        if not isinstance(source, list):
            continue

        chunks.extend(item for item in source if isinstance(item, str))

    return "".join(chunks)


def test_notebook_uses_nbformat_4() -> None:
    """Guard against accidental corruption of the historical notebook."""
    notebook = _load_notebook()

    assert notebook.get("nbformat") == 4


def test_notebook_contains_no_literal_quandl_api_key() -> None:
    """Prevent reintroduction of a literal Quandl API key assignment."""
    source = _code_source(_load_notebook())

    assert "quandl.ApiConfig.api_key" in source
    assert _LITERAL_API_KEY_ASSIGNMENT.search(source) is None
