"""Response integrity signals.

Signals are computed from submitted records alone and impose no additional
burden on respondents. Flags are deliberately kept separate from outcome
analysis so that screening can be applied while blinded to results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    """Screening thresholds, declared in advance rather than inherited."""

    seconds_per_item: float = 2.0
    duration_quantile: float = 0.05
    longstring_share: float = 1.0
    minimum_dispersion: float = 0.0
    origin_group_size: int = 3
    open_text_minimum_characters: int = 2


def longest_run(values: np.ndarray) -> int:
    """Longest sequence of identical adjacent responses."""
    if len(values) == 0:
        return 0
    best = run = 1
    for previous, current in zip(values[:-1], values[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def signals(frame: pd.DataFrame, items: list[str], open_text: list[str] | None = None) -> pd.DataFrame:
    """Compute one row of integrity signals per record."""
    responses = frame[items].to_numpy(dtype=float)
    out = pd.DataFrame(index=frame.index)

    out["duration_s"] = frame.get("duration_s", pd.Series(np.nan, index=frame.index))
    out["seconds_per_item"] = out["duration_s"] / len(items)
    out["longstring"] = [longest_run(row) for row in responses]
    out["longstring_share"] = out["longstring"] / len(items)
    out["dispersion"] = np.nanstd(responses, axis=1, ddof=1)
    out["mean_response"] = np.nanmean(responses, axis=1)

    centre = np.nanmean(responses, axis=0)
    spread = np.nanstd(responses, axis=0, ddof=1)
    spread[spread == 0] = np.nan
    out["mean_absolute_deviation"] = np.nanmean(np.abs((responses - centre) / spread), axis=1)

    if "origin" in frame:
        counts = frame["origin"].map(frame["origin"].value_counts())
        out["origin_group_size"] = counts
    else:
        out["origin_group_size"] = 1

    if open_text:
        text = frame[open_text].astype(str).replace({"nan": ""})
        out["open_text_characters"] = text.apply(lambda row: sum(len(v.strip()) for v in row), axis=1)
        out["open_text_fields_completed"] = text.apply(
            lambda row: sum(1 for v in row if len(v.strip()) > 0), axis=1
        )
    else:
        out["open_text_characters"] = np.nan
        out["open_text_fields_completed"] = np.nan

    return out


def flags(signal_frame: pd.DataFrame, thresholds: Thresholds, items_count: int) -> pd.DataFrame:
    """Convert signals into boolean flags under declared thresholds."""
    out = pd.DataFrame(index=signal_frame.index)
    duration_cut = signal_frame["duration_s"].quantile(thresholds.duration_quantile)

    out["flag_speed_absolute"] = signal_frame["seconds_per_item"] < thresholds.seconds_per_item
    out["flag_speed_relative"] = signal_frame["duration_s"] <= duration_cut
    out["flag_invariance"] = signal_frame["longstring_share"] >= thresholds.longstring_share
    out["flag_no_dispersion"] = signal_frame["dispersion"] <= thresholds.minimum_dispersion
    out["flag_origin_cluster"] = signal_frame["origin_group_size"] >= thresholds.origin_group_size
    out["flag_empty_text"] = signal_frame["open_text_characters"] <= thresholds.open_text_minimum_characters

    out["flag_count"] = out.sum(axis=1)
    out["screened_out"] = out["flag_count"] >= 2
    return out


def summarise(flag_frame: pd.DataFrame) -> pd.DataFrame:
    """Prevalence of each flag, for the audit trail."""
    counts = flag_frame.drop(columns=["flag_count", "screened_out"]).sum()
    share = counts / len(flag_frame) * 100
    return pd.DataFrame({"records": counts.astype(int), "percentage": share.round(1)})


def audit_log(frame: pd.DataFrame, flag_frame: pd.DataFrame, key: str = "record_key") -> pd.DataFrame:
    """Record-level decision log so that exclusions can be replicated exactly."""
    log = flag_frame.copy()
    log.insert(0, key, frame[key].to_numpy() if key in frame else frame.index)
    return log
