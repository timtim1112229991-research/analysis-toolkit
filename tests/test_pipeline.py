"""Tests exercising the pipeline on generated data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import comparison, integrity, reliability, sensitivity, simulate
from src.equivalence import Margin, equivalence_table, tost


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return simulate.demonstration_study(seed=7)


@pytest.fixture(scope="module")
def items(sample: pd.DataFrame) -> list[str]:
    return [c for c in sample.columns if c.startswith("item_")]


def test_generated_sample_is_well_formed(sample, items):
    assert len(items) == 12
    assert sample["arm"].nunique() == 2
    assert sample[items].to_numpy().min() >= 1
    assert sample[items].to_numpy().max() <= 5
    assert sample["contaminated"].sum() > 0


def test_alpha_is_bounded(sample, items):
    alpha = reliability.cronbach_alpha(sample[items])
    assert -1.0 <= alpha <= 1.0


def test_false_discovery_adjustment_is_monotone():
    raw = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    adjusted = comparison.benjamini_hochberg(raw)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)


def test_identical_arms_are_declared_equivalent():
    rng = np.random.default_rng(1)
    values = rng.normal(3.3, 0.6, 200)
    result = tost(values, values.copy(), margin=0.4)
    assert result["equivalent"] is True
    assert result["p_equivalence"] < 0.05


def test_separated_arms_are_not_declared_equivalent():
    rng = np.random.default_rng(2)
    a = rng.normal(4.2, 0.5, 120)
    b = rng.normal(3.0, 0.5, 120)
    result = tost(a, b, margin=0.4)
    assert result["equivalent"] is False


def test_equivalence_table_covers_every_measure(sample, items):
    margin = Margin(0.4, "scale points", "test fixture", "2026-09-02")
    table = equivalence_table(sample, items, margin)
    assert list(table.index) == items
    assert table["margin"].eq(0.4).all()


def test_longest_run_counts_adjacent_repeats():
    assert integrity.longest_run(np.array([3, 3, 3, 1, 2, 2])) == 3
    assert integrity.longest_run(np.array([1, 2, 3, 4])) == 1
    assert integrity.longest_run(np.array([])) == 0


def test_invariant_records_are_flagged(items):
    design = simulate.StudyDesign(n_per_arm=50)
    clean = simulate.clean_sample(design, seed=3)
    dirty = simulate.contaminate(clean, prevalence=0.2, archetype="invariant", seed=4)
    item_columns = [c for c in dirty.columns if c.startswith("item_")]

    computed = integrity.signals(dirty, item_columns)
    flagged = integrity.flags(computed, integrity.Thresholds(), len(item_columns))

    truly_invariant = dirty["archetype"].eq("invariant")
    detection = flagged.loc[truly_invariant, "flag_invariance"].mean()
    assert detection > 0.9
    false_flags = flagged.loc[~truly_invariant, "flag_invariance"].mean()
    assert false_flags < 0.2


def test_sensitivity_family_reports_multiple_severities(sample, items):
    computed = integrity.signals(sample, items)
    flagged = integrity.flags(computed, integrity.Thresholds(), len(items))
    sample = sample.assign(overall=sample[items].mean(axis=1))
    family = sensitivity.progressive_exclusion(sample, flagged, ["overall"])
    assert not family.empty
    assert family["severity"].nunique() >= 2
    assert family["retained"].is_monotonic_decreasing or family["retained"].nunique() > 1


def test_comparison_requires_two_arms(sample, items):
    single = sample[sample["arm"] == "control"]
    with pytest.raises(ValueError):
        comparison.compare(single, items)
