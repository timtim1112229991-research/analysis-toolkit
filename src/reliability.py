"""Internal consistency and item diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cronbach_alpha(frame: pd.DataFrame) -> float:
    """Coefficient alpha for a set of items scored in the same direction."""
    complete = frame.dropna()
    k = complete.shape[1]
    if k < 2 or len(complete) < 3:
        return float("nan")
    item_variance = complete.var(axis=0, ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_variance / total_variance))


def alpha_with_interval(frame: pd.DataFrame, draws: int = 2000, seed: int = 0) -> dict[str, float]:
    """Alpha with a bootstrap interval, which single point estimates omit."""
    rng = np.random.default_rng(seed)
    complete = frame.dropna()
    point = cronbach_alpha(complete)
    if not np.isfinite(point):
        return {"alpha": point, "lower": float("nan"), "upper": float("nan")}
    values = np.empty(draws)
    n = len(complete)
    for i in range(draws):
        sample = complete.iloc[rng.integers(0, n, n)]
        values[i] = cronbach_alpha(sample)
    lower, upper = np.nanpercentile(values, [2.5, 97.5])
    return {"alpha": point, "lower": float(lower), "upper": float(upper)}


def item_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Corrected item-total correlation and alpha if the item were dropped."""
    rows = []
    for item in frame.columns:
        others = [c for c in frame.columns if c != item]
        rest = frame[others].mean(axis=1)
        rows.append(
            {
                "item": item,
                "mean": float(frame[item].mean()),
                "sd": float(frame[item].std(ddof=1)),
                "corrected_item_total_r": float(frame[item].corr(rest)),
                "alpha_if_dropped": cronbach_alpha(frame[others]),
            }
        )
    return pd.DataFrame(rows)


def dimension_summary(frame: pd.DataFrame, dimensions: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, members in dimensions.items():
        stats = alpha_with_interval(frame[members])
        rows.append(
            {
                "dimension": name,
                "items": len(members),
                "alpha": stats["alpha"],
                "alpha_lower": stats["lower"],
                "alpha_upper": stats["upper"],
            }
        )
    return pd.DataFrame(rows)
