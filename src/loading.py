"""Assembly of a tidy analysis frame from positionally described workbooks."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Schema

_DURATION = re.compile(r"(\d+)")
_ORIGIN_DETAIL = re.compile(r"\((.+?)\)")


def _arm_for(filename: str, schema: Schema) -> str:
    for token, arm in schema.arm_from_filename.items():
        if token in filename:
            return arm
    return "unassigned"


def load_workbook(path: str | Path, schema: Schema) -> pd.DataFrame:
    """Read one workbook into the canonical column set."""
    path = Path(path)
    raw = pd.read_excel(path)
    cols = schema.columns
    coding = {label: rank + 1 for rank, label in enumerate(schema.ordinal_labels)}

    out = pd.DataFrame(index=raw.index)
    out["source"] = path.stem
    out["arm"] = _arm_for(path.name, schema)
    out["respondent_id"] = raw.iloc[:, cols.respondent_id]
    out["submitted_at"] = pd.to_datetime(raw.iloc[:, cols.submitted_at], errors="coerce")
    out["duration_s"] = (
        raw.iloc[:, cols.duration].astype(str).str.extract(_DURATION, expand=False).astype(float)
    )
    origin = raw.iloc[:, cols.origin].astype(str)
    out["origin"] = origin.str.split("(").str[0].str.strip()
    out["origin_detail"] = origin.str.extract(_ORIGIN_DETAIL, expand=False)

    if cols.reported_total is not None:
        out["reported_total"] = raw.iloc[:, cols.reported_total]

    for name, index in cols.demographics.items():
        out[name] = raw.iloc[:, index].astype(str).str.strip()

    for name, index in cols.items.items():
        out[name] = raw.iloc[:, index].astype(str).str.strip().map(coding)

    for name, index in cols.open_text.items():
        out[name] = raw.iloc[:, index]

    return out


def load_study(paths: list[str | Path], schema: Schema) -> pd.DataFrame:
    """Combine several workbooks and derive composites and wave labels."""
    schema.validate()
    frame = pd.concat([load_workbook(p, schema) for p in paths], ignore_index=True)

    items = schema.item_names()
    for dimension, members in schema.dimensions.items():
        frame[dimension] = frame[members].mean(axis=1)
    frame["overall"] = frame[items].mean(axis=1)
    frame["sum_score"] = frame[items].sum(axis=1)

    if schema.wave_boundary:
        boundary = pd.Timestamp(schema.wave_boundary)
        frame["wave"] = np.where(frame["submitted_at"] < boundary, "wave_1", "wave_2")
    else:
        frame["wave"] = "single"

    frame["record_key"] = frame["source"] + ":" + frame["respondent_id"].astype(str)
    return frame


def data_dictionary(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Describe every analysis variable for the audit trail."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "variable": column,
                "role": _role_of(column, schema),
                "dtype": str(series.dtype),
                "non_missing": int(series.notna().sum()),
                "distinct": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _role_of(column: str, schema: Schema) -> str:
    if column in schema.item_names():
        return "instrument item"
    if column in schema.dimensions:
        return "dimension composite"
    if column in schema.columns.demographics:
        return "respondent characteristic"
    if column in schema.columns.open_text:
        return "open response"
    if column in {"overall", "sum_score"}:
        return "composite"
    return "administrative"
