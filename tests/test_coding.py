"""Tests for the open response coding workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coding import (
    Code,
    Frame,
    agreement_report,
    build_sheet,
    cohen_kappa,
    disagreements,
    merge_resolved,
    pabak,
    theme_frequencies,
)

FRAME = Frame(
    field="comment",
    question="What stood out?",
    codes=(
        Code("SPEED", "Speed", "Mentions how quickly output arrived."),
        Code("DEPTH", "Depth", "Mentions thoroughness of the material."),
    ),
)


@pytest.fixture
def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_key": [f"r{i:02d}" for i in range(6)],
            "arm": ["treatment"] * 3 + ["control"] * 3,
            "comment": ["fast", "thorough", "fast and thorough", "slow", "", "brief"],
        }
    )


def test_sheet_is_blinded_and_empty(records):
    sheet = build_sheet(records, FRAME)
    assert "arm" not in sheet.columns
    assert list(sheet.columns) == ["record_key", "text", "SPEED", "DEPTH", "coder_note"]
    assert sheet[["SPEED", "DEPTH"]].isna().all().all()
    assert len(sheet) == 5, "empty responses are excluded from the sheet"


def test_sheet_shuffle_preserves_membership(records):
    ordered = build_sheet(records, FRAME)
    shuffled = build_sheet(records, FRAME, shuffle_seed=1)
    assert set(ordered["record_key"]) == set(shuffled["record_key"])


def test_missing_field_is_rejected(records):
    with pytest.raises(KeyError):
        build_sheet(records, Frame(field="absent", question="", codes=FRAME.codes))


def test_perfect_agreement_gives_unit_kappa():
    a = pd.Series([1, 0, 1, 0, 1, 0])
    assert cohen_kappa(a, a.copy()) == pytest.approx(1.0)


def test_complete_disagreement_gives_negative_kappa():
    a = pd.Series([1, 1, 0, 0])
    b = pd.Series([0, 0, 1, 1])
    assert cohen_kappa(a, b) < 0


def test_pabak_exceeds_kappa_for_a_rare_code():
    a = pd.Series([0] * 38 + [1, 0])
    b = pd.Series([0] * 38 + [0, 1])
    assert pabak(a, b) > cohen_kappa(a, b)


def test_agreement_report_covers_every_code(records):
    sheet = build_sheet(records, FRAME)
    coder_a = sheet.assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 1, 0, 0])
    coder_b = sheet.assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 0, 0, 0])
    report = agreement_report(coder_a, coder_b, FRAME)
    assert set(report.index) == {"SPEED", "DEPTH"}
    assert report.loc["SPEED", "percent_agreement"] == 1.0
    assert report.loc["DEPTH", "percent_agreement"] < 1.0


def test_disagreements_are_listed_for_adjudication(records):
    sheet = build_sheet(records, FRAME)
    coder_a = sheet.assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 1, 0, 0])
    coder_b = sheet.assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 0, 0, 0])
    disputed = disagreements(coder_a, coder_b, FRAME)
    assert len(disputed) == 1
    assert disputed.iloc[0]["code"] == "DEPTH"
    assert disputed.iloc[0]["resolution"] is pd.NA or pd.isna(disputed.iloc[0]["resolution"])


def test_resolution_is_applied_and_agreement_retained(records):
    sheet = build_sheet(records, FRAME)
    coder_a = sheet.assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 1, 0, 0])
    coder_b = sheet.assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 0, 0, 0])
    disputed = disagreements(coder_a, coder_b, FRAME)
    disputed["resolution"] = 1

    final = merge_resolved(coder_a, coder_b, disputed, FRAME)
    assert final.attrs["unresolved_cells"] == 0
    row = final[final["record_key"] == disputed.iloc[0]["record_key"]].iloc[0]
    assert row["DEPTH"] == 1
    assert row["SPEED"] == 1


def test_theme_frequencies_report_by_arm(records):
    sheet = build_sheet(records, FRAME)
    coded = sheet[["record_key"]].assign(SPEED=[1, 0, 1, 0, 0], DEPTH=[0, 1, 1, 0, 0])
    table = theme_frequencies(coded, records, FRAME)
    assert set(table.columns) == {"treatment", "control"}
    assert table.loc["SPEED", "treatment"] > table.loc["SPEED", "control"]
    assert np.isfinite(table.to_numpy()).all()
