"""Structured coding of open responses with two independent coders.

The workflow is deliberately conservative. Coders receive text stripped of any
arm or condition label, code independently against a fixed frame, and agreement
is computed before any theme frequency is reported. Disagreements are listed for
adjudication rather than resolved silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Code:
    """One category in a coding frame."""

    key: str
    label: str
    definition: str
    excludes: str = ""


@dataclass(frozen=True)
class Frame:
    """A coding frame applying to one open response field."""

    field: str
    question: str
    codes: tuple[Code, ...]
    multiple: bool = True

    def keys(self) -> list[str]:
        return [c.key for c in self.codes]

    def as_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"code": c.key, "label": c.label, "definition": c.definition, "excludes": c.excludes}
             for c in self.coding_order()]
        )

    def coding_order(self) -> tuple[Code, ...]:
        return self.codes


def build_sheet(
    records: pd.DataFrame,
    frame: Frame,
    key_column: str = "record_key",
    shuffle_seed: int | None = None,
) -> pd.DataFrame:
    """Produce a blinded coding sheet for one field.

    Arm, wave and every other design label are omitted so that a coder cannot
    infer the condition from the sheet.
    """
    if frame.field not in records:
        raise KeyError(f"field {frame.field} is absent from the records")

    sheet = pd.DataFrame(
        {
            key_column: records[key_column],
            "text": records[frame.field].astype(str).str.strip(),
        }
    )
    sheet = sheet[sheet["text"].str.len() > 0]

    if shuffle_seed is not None:
        sheet = sheet.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    for key in frame.keys():
        sheet[key] = pd.NA
    sheet["coder_note"] = pd.NA
    return sheet


def blind_keys(
    sheet: pd.DataFrame,
    key_column: str = "record_key",
    prefix: str = "C",
    digits: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace record keys with opaque identifiers and return the crosswalk.

    Record keys derived from source filenames frequently carry the condition in
    readable form, which would defeat the point of a blinded sheet. The
    crosswalk is held by the analyst and is not given to coders.
    """
    if key_column not in sheet:
        raise KeyError(f"column {key_column} is absent from the sheet")

    blinded = sheet.copy()
    ids = [f"{prefix}{i + 1:0{digits}d}" for i in range(len(blinded))]
    crosswalk = pd.DataFrame({"coding_id": ids, key_column: blinded[key_column].to_numpy()})

    blinded.insert(0, "coding_id", ids)
    blinded = blinded.drop(columns=[key_column])
    return blinded, crosswalk


def restore_keys(
    sheet: pd.DataFrame, crosswalk: pd.DataFrame, key_column: str = "record_key"
) -> pd.DataFrame:
    """Rejoin true record keys to a completed sheet."""
    merged = sheet.merge(crosswalk, on="coding_id", how="left")
    if merged[key_column].isna().any():
        raise ValueError("the crosswalk does not cover every coded record")
    return merged.drop(columns=["coding_id"])


def percent_agreement(a: pd.Series, b: pd.Series) -> float:
    paired = pd.concat([a, b], axis=1).dropna()
    if paired.empty:
        return float("nan")
    return float((paired.iloc[:, 0] == paired.iloc[:, 1]).mean())


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected agreement for two coders on one code."""
    paired = pd.concat([a, b], axis=1).dropna()
    if paired.empty:
        return float("nan")
    left, right = paired.iloc[:, 0], paired.iloc[:, 1]
    categories = sorted(set(left) | set(right))
    if len(categories) < 2:
        return float("nan")
    matrix = pd.crosstab(left, right).reindex(index=categories, columns=categories, fill_value=0)
    total = matrix.to_numpy().sum()
    observed = np.trace(matrix.to_numpy()) / total
    expected = (matrix.sum(axis=0).to_numpy() * matrix.sum(axis=1).to_numpy()).sum() / total**2
    if expected >= 1:
        return float("nan")
    return float((observed - expected) / (1 - expected))


def pabak(a: pd.Series, b: pd.Series) -> float:
    """Prevalence adjusted agreement, which is informative for rare codes.

    Coefficient kappa collapses towards zero when a code is very rare even
    though the coders agree on almost every record, so both are reported.
    """
    observed = percent_agreement(a, b)
    if not np.isfinite(observed):
        return float("nan")
    return float(2 * observed - 1)


def krippendorff_alpha_nominal(a: pd.Series, b: pd.Series) -> float:
    """Alpha for two coders on a nominal or binary code."""
    paired = pd.concat([a, b], axis=1).dropna()
    if paired.empty:
        return float("nan")
    values = paired.to_numpy()
    n_units = len(values)
    categories = sorted(set(values.ravel()))
    if len(categories) < 2:
        return float("nan")

    disagreement = float((values[:, 0] != values[:, 1]).sum())
    observed = disagreement / n_units

    flat = values.ravel()
    counts = np.array([(flat == c).sum() for c in categories], dtype=float)
    total = counts.sum()
    expected = 1 - (counts * (counts - 1)).sum() / (total * (total - 1))
    if expected == 0:
        return float("nan")
    return float(1 - observed / expected)


def agreement_report(
    coder_a: pd.DataFrame, coder_b: pd.DataFrame, frame: Frame, key_column: str = "record_key"
) -> pd.DataFrame:
    """Per-code agreement between two completed sheets."""
    left = coder_a.set_index(key_column)
    right = coder_b.set_index(key_column)
    shared = left.index.intersection(right.index)
    if len(shared) == 0:
        raise ValueError("the two sheets share no records")

    rows = []
    for key in frame.keys():
        a = left.loc[shared, key].astype("Int64")
        b = right.loc[shared, key].astype("Int64")
        applied = int(((a == 1) | (b == 1)).sum())
        rows.append(
            {
                "code": key,
                "records": len(shared),
                "times_applied": applied,
                "prevalence": round(applied / len(shared), 3),
                "percent_agreement": round(percent_agreement(a, b), 3),
                "cohen_kappa": round(cohen_kappa(a, b), 3),
                "pabak": round(pabak(a, b), 3),
                "krippendorff_alpha": round(krippendorff_alpha_nominal(a, b), 3),
            }
        )
    return pd.DataFrame(rows).set_index("code")


def disagreements(
    coder_a: pd.DataFrame, coder_b: pd.DataFrame, frame: Frame, key_column: str = "record_key"
) -> pd.DataFrame:
    """Records requiring adjudication, one row per disputed code."""
    left = coder_a.set_index(key_column)
    right = coder_b.set_index(key_column)
    shared = left.index.intersection(right.index)

    rows = []
    for key in frame.keys():
        a = left.loc[shared, key]
        b = right.loc[shared, key]
        differing = shared[(a.fillna(-1) != b.fillna(-1)).to_numpy()]
        for record in differing:
            rows.append(
                {
                    key_column: record,
                    "code": key,
                    "coder_a": left.loc[record, key],
                    "coder_b": right.loc[record, key],
                    "text": left.loc[record, "text"] if "text" in left else "",
                    "resolution": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def merge_resolved(
    coder_a: pd.DataFrame, coder_b: pd.DataFrame, resolved: pd.DataFrame, frame: Frame,
    key_column: str = "record_key",
) -> pd.DataFrame:
    """Final coding: agreed codes retained, disputed codes taken from adjudication."""
    left = coder_a.set_index(key_column)
    right = coder_b.set_index(key_column)
    shared = left.index.intersection(right.index)
    final = pd.DataFrame(index=shared)

    for key in frame.keys():
        a = left.loc[shared, key]
        b = right.loc[shared, key]
        agreed = a.where(a.fillna(-1) == b.fillna(-1))
        final[key] = agreed

    if not resolved.empty:
        for _, row in resolved.iterrows():
            if pd.notna(row.get("resolution")):
                final.loc[row[key_column], row["code"]] = row["resolution"]

    unresolved = int(final.isna().sum().sum())
    final.attrs["unresolved_cells"] = unresolved
    return final.reset_index()


def theme_frequencies(
    coded: pd.DataFrame, design: pd.DataFrame, frame: Frame, group: str = "arm",
    key_column: str = "record_key",
) -> pd.DataFrame:
    """Theme prevalence by group, joined to the design after coding is closed."""
    merged = coded.merge(design[[key_column, group]], on=key_column, how="left")
    rows = []
    for key in frame.keys():
        for value, block in merged.groupby(group):
            applied = int((block[key] == 1).sum())
            rows.append(
                {
                    "code": key,
                    group: value,
                    "records": len(block),
                    "applied": applied,
                    "percentage": round(100 * applied / len(block), 1) if len(block) else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    return table.pivot(index="code", columns=group, values="percentage")
