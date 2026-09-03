"""Inference when observations share a cluster and are therefore not independent.

Field studies routinely recruit through institutions, so respondents arrive in
groups that share a network origin, a workplace and often a conversation about
the thing being evaluated. Treating such records as independent understates the
standard error of a between-arm contrast. Where a cluster contributes to both
arms it is stronger still: the same person may have supplied both observations.

Two remedies are provided. The cluster bootstrap resamples whole clusters and
makes no assumption about who is paired with whom. The matched pair analysis
uses only clusters contributing exactly one record to each arm, which is a
smaller but cleanly paired sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

#: Resampling seed used unless a caller supplies its own. Named here rather
#: than repeated in each signature so that a run manifest can cite the same
#: value the estimator actually used.
DEFAULT_SEED = 20260902


@dataclass(frozen=True)
class ClusterDiagnosis:
    """Summary of how far a design departs from independent sampling."""

    clusters: int
    records: int
    crossing_clusters: int
    records_in_crossing_clusters: int
    largest_cluster: int
    matched_pairs: int

    @property
    def crossing_share(self) -> float:
        return self.records_in_crossing_clusters / self.records if self.records else float("nan")

    def as_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("distinct clusters", self.clusters),
                ("records", self.records),
                ("clusters appearing in both arms", self.crossing_clusters),
                ("records in a crossing cluster", self.records_in_crossing_clusters),
                ("share of records in a crossing cluster", round(self.crossing_share, 3)),
                ("largest cluster", self.largest_cluster),
                ("clusters forming a one to one pair", self.matched_pairs),
            ],
            columns=["quantity", "value"],
        ).set_index("quantity")


def diagnose(frame: pd.DataFrame, group: str, cluster: str) -> ClusterDiagnosis:
    """Quantify the departure from independent sampling."""
    counts = frame.groupby(cluster)[group].nunique()
    crossing = counts[counts > 1].index
    sizes = frame.groupby(cluster).size()

    per_arm = frame.groupby([cluster, group]).size().unstack(fill_value=0)
    paired = int(((per_arm == 1).all(axis=1) & (per_arm.shape[1] == 2)).sum()) if not per_arm.empty else 0

    return ClusterDiagnosis(
        clusters=int(frame[cluster].nunique()),
        records=int(len(frame)),
        crossing_clusters=int(len(crossing)),
        records_in_crossing_clusters=int(frame[cluster].isin(crossing).sum()),
        largest_cluster=int(sizes.max()) if len(sizes) else 0,
        matched_pairs=paired,
    )


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    value: str,
    group: str,
    cluster: str,
    arms: tuple[str, str],
    n_boot: int = 5000,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean difference between arms with a cluster bootstrap interval.

    Whole clusters are resampled with replacement, so any correlation within a
    cluster, including the extreme case of one person answering twice, is
    carried into the interval rather than assumed away.
    """
    work = frame[[value, group, cluster]].dropna()
    work = work[work[group].isin(arms)]
    if work.empty:
        raise ValueError("no records remain for this contrast")

    def difference(sample: pd.DataFrame) -> float:
        left = sample.loc[sample[group] == arms[0], value]
        right = sample.loc[sample[group] == arms[1], value]
        if left.empty or right.empty:
            return np.nan
        return float(left.mean() - right.mean())

    observed = difference(work)
    keys = work[cluster].unique()
    blocks = {key: block for key, block in work.groupby(cluster)}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        resample = pd.concat([blocks[key] for key in chosen], ignore_index=True)
        draws[i] = difference(resample)

    draws = draws[np.isfinite(draws)]
    lower, upper = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    naive = stats.ttest_ind(
        work.loc[work[group] == arms[0], value],
        work.loc[work[group] == arms[1], value],
        equal_var=False,
    )

    return {
        "difference": round(observed, 4),
        "cluster_se": round(float(np.std(draws, ddof=1)), 4),
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "clusters": len(keys),
        "records": len(work),
        "naive_p": round(float(naive.pvalue), 4),
        "resamples": len(draws),
    }


def cluster_equivalence(
    frame: pd.DataFrame,
    value: str,
    group: str,
    cluster: str,
    arms: tuple[str, str],
    margin: float,
    n_boot: int = 5000,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float | bool | str]:
    """Equivalence by interval inclusion, using a cluster bootstrap interval.

    The two one-sided tests are equivalent to asking whether the interval at
    confidence one minus two alpha lies wholly inside the margin, which is the
    form used here because the bootstrap supplies an interval directly.
    """
    result = cluster_bootstrap_difference(
        frame, value, group, cluster, arms, n_boot=n_boot, seed=seed, alpha=alpha
    )
    inside = bool(result["lower"] > -margin and result["upper"] < margin)
    straddles = bool(result["lower"] < -margin or result["upper"] > margin)
    result.update(
        {
            "margin": margin,
            "equivalent": inside,
            "verdict": "equivalent" if inside else ("different" if not straddles else "inconclusive"),
        }
    )
    return result


def matched_pairs(
    frame: pd.DataFrame, group: str, cluster: str, arms: tuple[str, str]
) -> pd.DataFrame:
    """Clusters contributing exactly one record to each arm, returned wide.

    This is the cleanest reading of the data available when respondent
    identifiers were never recorded: a cluster of size one in each arm is most
    plausibly a single person seen twice.
    """
    work = frame[frame[group].isin(arms)]
    counts = work.groupby([cluster, group]).size().unstack(fill_value=0)
    eligible = counts[(counts.get(arms[0], 0) == 1) & (counts.get(arms[1], 0) == 1)].index
    subset = work[work[cluster].isin(eligible)]
    return subset.pivot_table(index=cluster, columns=group, aggfunc="first")


def paired_contrast(
    frame: pd.DataFrame,
    value: str,
    group: str,
    cluster: str,
    arms: tuple[str, str],
    margin: float | None = None,
    alpha: float = 0.05,
) -> dict[str, float | bool | int | str]:
    """Paired comparison over one to one clusters, with optional equivalence."""
    work = frame[frame[group].isin(arms)][[cluster, group, value]].dropna()
    counts = work.groupby([cluster, group]).size().unstack(fill_value=0)
    eligible = counts[(counts.get(arms[0], 0) == 1) & (counts.get(arms[1], 0) == 1)].index
    wide = (
        work[work[cluster].isin(eligible)]
        .pivot(index=cluster, columns=group, values=value)
        .dropna()
    )
    if len(wide) < 3:
        return {"pairs": len(wide), "verdict": "too few pairs"}

    left, right = wide[arms[0]], wide[arms[1]]
    differences = left - right
    n = len(differences)
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(n))

    if differences.nunique() == 1:
        signed_p = float("nan")
    else:
        signed_p = float(stats.wilcoxon(left, right, zero_method="zsplit").pvalue)

    critical = stats.t.ppf(1 - alpha, df=n - 1)
    out: dict[str, float | bool | int | str] = {
        "pairs": n,
        "mean_difference": round(mean, 4),
        "se": round(se, 4),
        "lower": round(mean - critical * se, 4),
        "upper": round(mean + critical * se, 4),
        "wilcoxon_p": round(signed_p, 4) if np.isfinite(signed_p) else float("nan"),
        "paired_t_p": round(float(stats.ttest_rel(left, right).pvalue), 4),
    }
    if margin is not None:
        inside = bool(out["lower"] > -margin and out["upper"] < margin)
        out["margin"] = margin
        out["equivalent"] = inside
        out["verdict"] = "equivalent" if inside else "inconclusive"
    return out


def cluster_bootstrap_coefficients(
    frame: pd.DataFrame,
    cluster: str,
    estimator: Callable[[pd.DataFrame], pd.Series],
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    exponentiate: bool = False,
) -> pd.DataFrame:
    """Bootstrap any coefficient vector by resampling whole clusters.

    Models without a clustered covariance option, the proportional odds model
    among them, can still be reported honestly by refitting on cluster
    resamples. Fits that fail to converge on a resample are discarded and
    counted, since a silent drop would bias the interval.
    """
    observed = estimator(frame)
    keys = frame[cluster].unique()
    blocks = {key: block for key, block in frame.groupby(cluster)}

    rng = np.random.default_rng(seed)
    draws: list[pd.Series] = []
    failures = 0
    for _ in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        resample = pd.concat([blocks[key] for key in chosen], ignore_index=True)
        try:
            draws.append(estimator(resample))
        except Exception:
            failures += 1

    if not draws:
        raise RuntimeError("every bootstrap fit failed")

    matrix = pd.DataFrame(draws).reindex(columns=observed.index)
    lower = matrix.quantile(alpha / 2)
    upper = matrix.quantile(1 - alpha / 2)

    table = pd.DataFrame(
        {
            "coefficient": observed.round(4),
            "cluster_se": matrix.std(ddof=1).round(4),
            "lower": lower.round(4),
            "upper": upper.round(4),
        }
    )
    table["excludes_zero"] = (table["lower"] > 0) | (table["upper"] < 0)
    if exponentiate:
        for column in ("coefficient", "lower", "upper"):
            table[f"or_{column}"] = np.exp(table[column]).round(3)

    table.attrs["resamples"] = len(draws)
    table.attrs["failures"] = failures
    table.attrs["clusters"] = len(keys)
    return table


def cluster_robust_linear(
    frame: pd.DataFrame, outcome: str, predictors: list[str], cluster: str
):
    """Least squares with standard errors clustered on the recruiting unit."""
    import statsmodels.api as sm

    work = frame[[outcome, *predictors, cluster]].dropna()
    design = sm.add_constant(work[predictors].astype(float))
    model = sm.OLS(work[outcome].astype(float), design)
    return model.fit(cov_type="cluster", cov_kwds={"groups": work[cluster]})


def summarise_fit(fit) -> pd.DataFrame:
    """Coefficients, clustered intervals and probabilities in one table."""
    table = pd.DataFrame(
        {
            "coefficient": fit.params.round(4),
            "std_error": fit.bse.round(4),
            "p_value": fit.pvalues.round(4),
        }
    )
    intervals = fit.conf_int()
    table["lower"] = intervals[0].round(4)
    table["upper"] = intervals[1].round(4)
    return table
