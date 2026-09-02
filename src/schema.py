"""Schema description and discovery.

Source workbooks are described by position rather than by header text, so that
no study-specific or source-language content enters the code base. A schema is
generated locally from a workbook, completed by the analyst, and kept out of
version control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ColumnMap:
    """Positional description of a source workbook."""

    respondent_id: int
    submitted_at: int
    duration: int
    origin: int
    reported_total: int | None = None
    demographics: dict[str, int] = field(default_factory=dict)
    items: dict[str, int] = field(default_factory=dict)
    open_text: dict[str, int] = field(default_factory=dict)


@dataclass
class Schema:
    """Full description of a study's source files and measurement model."""

    columns: ColumnMap
    ordinal_labels: list[str]
    dimensions: dict[str, list[str]]
    arm_from_filename: dict[str, str] = field(default_factory=dict)
    wave_boundary: str | None = None

    def to_json(self, path: str | Path) -> None:
        payload: dict[str, Any] = asdict(self)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Schema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["columns"] = ColumnMap(**payload["columns"])
        return cls(**payload)

    def item_names(self) -> list[str]:
        return list(self.columns.items.keys())

    def validate(self) -> None:
        known = set(self.item_names())
        for name, members in self.dimensions.items():
            unknown = set(members) - known
            if unknown:
                raise ValueError(f"dimension {name} references unknown items: {sorted(unknown)}")
        if len(self.ordinal_labels) < 2:
            raise ValueError("at least two ordinal labels are required")
        if len(set(self.ordinal_labels)) != len(self.ordinal_labels):
            raise ValueError("ordinal labels must be unique")


def discover(workbook: str | Path, item_first: int, item_last: int) -> dict[str, Any]:
    """Inspect a workbook and return a schema template for manual completion.

    The template lists every column by index together with the distinct values
    found in the candidate item range. The analyst supplies the ordering of the
    ordinal labels, which cannot be inferred reliably from frequencies alone.
    """
    frame = pd.read_excel(workbook)
    columns = [{"index": i, "distinct": int(frame.iloc[:, i].nunique(dropna=True))}
               for i in range(frame.shape[1])]

    observed: list[str] = []
    for i in range(item_first, item_last + 1):
        for value in frame.iloc[:, i].dropna().astype(str).str.strip().unique():
            if value not in observed:
                observed.append(value)

    return {
        "columns_by_index": columns,
        "observed_item_values": observed,
        "note": "Order observed_item_values from lowest to highest before use.",
    }
