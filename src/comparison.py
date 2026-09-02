"""Two-arm comparison with effect sizes, intervals and multiplicity control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _clean(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def cohen_d(a, b) -> float:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled else float("nan")


def rank_biserial(a, b) -> float:
    """Effect size for the rank-based test, bounded in the interval -1 to 1."""
    a, b = _clean(a), _clean(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def difference_interval(a, b, draws: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap interval for the difference in means."""
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    for i in range(draws):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(lower), float(upper)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Step-up false discovery rate adjustment."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank, index in enumerate(order[::-1]):
        running = min(running, p[index] * m / (m - rank))
        adjusted[index] = running
    return adjusted


def compare(
    frame: pd.DataFrame,
    measures: list[str],
    arm_column: str = "arm",
    reference: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare two arms across a family of measures."""
    arms = sorted(frame[arm_column].dropna().unique())
    if len(arms) != 2:
        raise ValueError(f"expected exactly two arms, found {arms}")
    if reference is not None:
        arms = [a for a in arms if a != reference] + [reference]
    treated, control = arms

    rows = []
    for measure in measures:
        a = frame.loc[frame[arm_column] == treated, measure]
        b = frame.loc[frame[arm_column] == control, measure]
        a_clean, b_clean = _clean(a), _clean(b)
        welch_t, welch_p = stats.ttest_ind(a_clean, b_clean, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
        lower, upper = difference_interval(a_clean, b_clean, seed=seed)
        rows.append(
            {
                "measure": measure,
                f"n_{treated}": len(a_clean),
                f"mean_{treated}": a_clean.mean(),
                f"sd_{treated}": a_clean.std(ddof=1),
                f"n_{control}": len(b_clean),
                f"mean_{control}": b_clean.mean(),
                f"sd_{control}": b_clean.std(ddof=1),
                "difference": a_clean.mean() - b_clean.mean(),
                "difference_lower": lower,
                "difference_upper": upper,
                "welch_t": welch_t,
                "welch_p": welch_p,
                "mann_whitney_u": u_stat,
                "mann_whitney_p": u_p,
                "cohen_d": cohen_d(a_clean, b_clean),
                "rank_biserial": rank_biserial(a_clean, b_clean),
            }
        )

    result = pd.DataFrame(rows).set_index("measure")
    result["mann_whitney_p_adjusted"] = benjamini_hochberg(result["mann_whitney_p"].to_numpy())
    return result


def response_distribution(frame: pd.DataFrame, item: str, arm_column: str = "arm") -> pd.DataFrame:
    """Percentage of respondents at each scale point, by arm."""
    table = pd.crosstab(frame[arm_column], frame[item], normalize="index") * 100
    return table.round(1)


def top_box(frame: pd.DataFrame, items: list[str], threshold: int, arm_column: str = "arm") -> pd.DataFrame:
    """Share of responses at or above a scale point, by arm."""
    parts = {arm: (group[items] >= threshold).mean() * 100 for arm, group in frame.groupby(arm_column)}
    table = pd.DataFrame(parts)
    if table.shape[1] == 2:
        left, right = table.columns
        table["difference"] = table[left] - table[right]
    return table.round(1)
