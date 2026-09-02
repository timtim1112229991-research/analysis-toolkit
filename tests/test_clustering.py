"""Tests for inference under clustered sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    cluster_bootstrap_coefficients,
    cluster_bootstrap_difference,
    cluster_equivalence,
    cluster_robust_linear,
    diagnose,
    matched_pairs,
    paired_contrast,
    summarise_fit,
)

ARMS = ("treatment", "control")


@pytest.fixture
def crossed() -> pd.DataFrame:
    """Twelve clusters, each contributing one record to each arm."""
    rng = np.random.default_rng(7)
    rows = []
    for c in range(12):
        level = rng.normal(3.5, 0.8)
        for arm in ARMS:
            rows.append({"origin": f"o{c}", "arm": arm, "score": level + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


@pytest.fixture
def independent() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for c in range(24):
        arm = ARMS[c % 2]
        rows.append({"origin": f"o{c}", "arm": arm, "score": rng.normal(3.5, 0.8)})
    return pd.DataFrame(rows)


def test_diagnosis_detects_full_crossing(crossed):
    report = diagnose(crossed, "arm", "origin")
    assert report.clusters == 12
    assert report.crossing_clusters == 12
    assert report.crossing_share == 1.0
    assert report.matched_pairs == 12


def test_diagnosis_detects_clean_separation(independent):
    report = diagnose(independent, "arm", "origin")
    assert report.crossing_clusters == 0
    assert report.matched_pairs == 0
    assert report.crossing_share == 0.0


def test_diagnosis_table_is_readable(crossed):
    table = diagnose(crossed, "arm", "origin").as_table()
    assert "records" in table.index
    assert table.loc["records", "value"] == 24


def test_cluster_bootstrap_returns_a_bounded_interval(crossed):
    result = cluster_bootstrap_difference(crossed, "score", "arm", "origin", ARMS, n_boot=400)
    assert result["lower"] < result["difference"] < result["upper"]
    assert result["clusters"] == 12
    assert result["records"] == 24


def test_clustering_is_reflected_in_the_interval(crossed):
    """Within cluster correlation is strong here, so the clustered interval is
    narrower than the naive test implies, and the two must not coincide."""
    result = cluster_bootstrap_difference(crossed, "score", "arm", "origin", ARMS, n_boot=800)
    naive_se = 0.8 * np.sqrt(2 / 12)
    assert result["cluster_se"] < naive_se


def test_bootstrap_is_reproducible(crossed):
    first = cluster_bootstrap_difference(crossed, "score", "arm", "origin", ARMS, n_boot=300, seed=3)
    second = cluster_bootstrap_difference(crossed, "score", "arm", "origin", ARMS, n_boot=300, seed=3)
    assert first == second


def test_equivalence_verdict_respects_the_margin(crossed):
    wide = cluster_equivalence(crossed, "score", "arm", "origin", ARMS, margin=1.0, n_boot=400)
    narrow = cluster_equivalence(crossed, "score", "arm", "origin", ARMS, margin=0.001, n_boot=400)
    assert wide["equivalent"] is True
    assert narrow["equivalent"] is False
    assert narrow["verdict"] in {"inconclusive", "different"}


def test_empty_contrast_is_rejected(crossed):
    with pytest.raises(ValueError):
        cluster_bootstrap_difference(crossed, "score", "arm", "origin", ("absent", "missing"))


def test_matched_pairs_selects_one_to_one_clusters(crossed):
    extra = pd.concat(
        [crossed, pd.DataFrame([{"origin": "o0", "arm": "treatment", "score": 4.0}])],
        ignore_index=True,
    )
    pairs = matched_pairs(extra, "arm", "origin", ARMS)
    assert "o0" not in pairs.index, "a cluster with two records in one arm is not a pair"
    assert len(pairs) == 11


def test_paired_contrast_uses_only_pairs(crossed):
    result = paired_contrast(crossed, "score", "arm", "origin", ARMS, margin=1.0)
    assert result["pairs"] == 12
    assert result["equivalent"] is True
    assert abs(result["mean_difference"]) < 1.0


def test_paired_contrast_declines_when_pairs_are_scarce(independent):
    result = paired_contrast(independent, "score", "arm", "origin", ARMS)
    assert result["verdict"] == "too few pairs"


def _slope(frame: pd.DataFrame) -> pd.Series:
    import statsmodels.api as sm

    design = sm.add_constant(frame[["treated"]].astype(float))
    return sm.OLS(frame["score"].astype(float), design).fit().params


def test_bootstrap_coefficients_bracket_the_estimate(crossed):
    frame = crossed.assign(treated=(crossed["arm"] == "treatment").astype(int))
    table = cluster_bootstrap_coefficients(frame, "origin", _slope, n_boot=300)
    assert set(table.index) == {"const", "treated"}
    assert (table["lower"] <= table["coefficient"]).all()
    assert (table["upper"] >= table["coefficient"]).all()
    assert table.attrs["resamples"] == 300
    assert table.attrs["failures"] == 0


def test_bootstrap_coefficients_can_report_odds_ratios(crossed):
    frame = crossed.assign(treated=(crossed["arm"] == "treatment").astype(int))
    table = cluster_bootstrap_coefficients(
        frame, "origin", _slope, n_boot=200, exponentiate=True
    )
    assert "or_coefficient" in table.columns
    assert (table["or_coefficient"] > 0).all()


def test_failed_fits_are_counted_not_hidden(crossed):
    frame = crossed.assign(treated=(crossed["arm"] == "treatment").astype(int))
    calls = {"n": 0}

    def flaky(sample: pd.DataFrame) -> pd.Series:
        calls["n"] += 1
        if calls["n"] % 3 == 0 and calls["n"] > 1:
            raise RuntimeError("did not converge")
        return _slope(sample)

    table = cluster_bootstrap_coefficients(frame, "origin", flaky, n_boot=90)
    assert table.attrs["failures"] > 0
    assert table.attrs["resamples"] + table.attrs["failures"] == 90


def test_cluster_robust_fit_reports_intervals(crossed):
    frame = crossed.assign(treated=(crossed["arm"] == "treatment").astype(int))
    fit = cluster_robust_linear(frame, "score", ["treated"], "origin")
    table = summarise_fit(fit)
    assert set(table.columns) == {"coefficient", "std_error", "p_value", "lower", "upper"}
    assert (table["lower"] <= table["coefficient"]).all()
    assert (table["upper"] >= table["coefficient"]).all()
