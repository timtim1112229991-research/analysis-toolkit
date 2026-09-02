"""Equivalence testing.

A non-significant difference does not establish similarity. These routines
implement two one-sided tests against a margin that must be fixed before the
data are analysed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Margin:
    """Equivalence margin with the justification recorded alongside it."""

    value: float
    unit: str
    justification: str
    declared_on: str

    def as_row(self) -> dict[str, object]:
        return {
            "margin": self.value,
            "unit": self.unit,
            "justification": self.justification,
            "declared_on": self.declared_on,
        }


def tost(a, b, margin: float, alpha: float = 0.05) -> dict[str, float | bool]:
    """Two one-sided tests for the difference in means on a raw scale."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each arm needs at least two observations")

    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se**4 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )

    t_lower = (diff + margin) / se
    t_upper = (diff - margin) / se
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)

    half_width = stats.t.ppf(1 - alpha, df) * se
    return {
        "difference": float(diff),
        "standard_error": float(se),
        "df": float(df),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_equivalence": float(p_tost),
        "ci_lower": float(diff - half_width),
        "ci_upper": float(diff + half_width),
        "equivalent": bool(p_tost < alpha),
    }


def equivalence_table(
    frame: pd.DataFrame,
    measures: list[str],
    margin: Margin,
    arm_column: str = "arm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply the two one-sided tests across a family of measures."""
    arms = sorted(frame[arm_column].dropna().unique())
    if len(arms) != 2:
        raise ValueError(f"expected exactly two arms, found {arms}")
    treated, control = arms

    rows = []
    for measure in measures:
        a = frame.loc[frame[arm_column] == treated, measure]
        b = frame.loc[frame[arm_column] == control, measure]
        result = tost(a, b, margin.value, alpha=alpha)
        result["measure"] = measure
        rows.append(result)

    table = pd.DataFrame(rows).set_index("measure")
    table["margin"] = margin.value
    return table


def smallest_detectable_difference(
    frame: pd.DataFrame, measure: str, arm_column: str = "arm", power: float = 0.80, alpha: float = 0.05
) -> float:
    """Difference the design can resolve, which separates a null from a weak test."""
    groups = [np.asarray(g[measure].dropna(), dtype=float) for _, g in frame.groupby(arm_column)]
    if len(groups) != 2:
        raise ValueError("two arms required")
    a, b = groups
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * pooled * np.sqrt(1 / len(a) + 1 / len(b)))
