"""Sensitivity of conclusions to screening severity.

Reporting a single post-exclusion estimate hides how much the conclusion
depends on the screening rule. These routines produce the whole family.
"""

from __future__ import annotations

import pandas as pd

from .comparison import compare
from .equivalence import Margin, equivalence_table


def progressive_exclusion(
    frame: pd.DataFrame,
    flag_frame: pd.DataFrame,
    measures: list[str],
    margin: Margin | None = None,
    arm_column: str = "arm",
    max_severity: int | None = None,
) -> pd.DataFrame:
    """Recompute comparisons as records are removed at increasing severity.

    Severity zero retains every record. Severity k removes records carrying k or
    more integrity flags.
    """
    counts = flag_frame["flag_count"]
    ceiling = int(counts.max()) if max_severity is None else max_severity

    blocks = []
    for severity in range(ceiling + 1):
        keep = counts < severity if severity > 0 else pd.Series(True, index=frame.index)
        subset = frame.loc[keep.to_numpy()]
        if subset[arm_column].nunique() != 2 or len(subset) < 8:
            continue
        block = compare(subset, measures, arm_column=arm_column)
        block = block[["difference", "difference_lower", "difference_upper",
                       "mann_whitney_p", "mann_whitney_p_adjusted", "cohen_d"]]
        block.insert(0, "severity", severity)
        block.insert(1, "retained", len(subset))
        if margin is not None:
            eq = equivalence_table(subset, measures, margin, arm_column=arm_column)
            block["p_equivalence"] = eq["p_equivalence"]
            block["equivalent"] = eq["equivalent"]
        blocks.append(block.reset_index())

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def conclusion_stability(sensitivity: pd.DataFrame, column: str = "equivalent") -> pd.DataFrame:
    """Report whether the conclusion holds across every severity level."""
    if sensitivity.empty or column not in sensitivity:
        return pd.DataFrame()
    grouped = sensitivity.groupby("measure")[column]
    return pd.DataFrame(
        {
            "levels": grouped.size(),
            "holds_everywhere": grouped.all(),
            "holds_anywhere": grouped.any(),
        }
    )
