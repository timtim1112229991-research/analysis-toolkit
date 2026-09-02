"""Language-neutral descriptors for open responses.

Coding of meaning is performed by human coders. What is automated here is
limited to substance measurement and agreement statistics, neither of which
depends on the language of the response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def substance(frame: pd.DataFrame, fields: list[str], arm_column: str = "arm") -> pd.DataFrame:
    """Length and completion descriptors per arm, used as an effort indicator."""
    rows = []
    for arm, group in frame.groupby(arm_column):
        for field in fields:
            text = group[field].astype(str).replace({"nan": ""}).str.strip()
            answered = text[text.str.len() > 0]
            rows.append(
                {
                    "arm": arm,
                    "field": field,
                    "records": len(text),
                    "answered": len(answered),
                    "answer_rate": round(100 * len(answered) / len(text), 1) if len(text) else np.nan,
                    "median_characters": float(answered.str.len().median()) if len(answered) else np.nan,
                    "distinct_answers": int(answered.nunique()),
                }
            )
    return pd.DataFrame(rows)


def duplicate_responses(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Identical open responses, which indicate either boilerplate or copying."""
    rows = []
    for field in fields:
        text = frame[field].astype(str).str.strip()
        text = text[(text.str.len() > 0) & (text != "nan")]
        counts = text.value_counts()
        repeated = counts[counts > 1]
        rows.append(
            {
                "field": field,
                "answers": int(len(text)),
                "distinct": int(counts.size),
                "repeated_values": int(repeated.size),
                "records_in_repeats": int(repeated.sum()),
            }
        )
    return pd.DataFrame(rows)


def cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    """Agreement between two coders, corrected for chance."""
    paired = pd.concat([first, second], axis=1).dropna()
    if paired.empty:
        return float("nan")
    a, b = paired.iloc[:, 0], paired.iloc[:, 1]
    categories = sorted(set(a) | set(b))
    matrix = pd.crosstab(a, b).reindex(index=categories, columns=categories, fill_value=0)
    total = matrix.to_numpy().sum()
    observed = np.trace(matrix.to_numpy()) / total
    expected = (matrix.sum(axis=0).to_numpy() * matrix.sum(axis=1).to_numpy()).sum() / total**2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")
