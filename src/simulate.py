"""Synthetic data generation.

Two uses. First, the repository ships no real records, so demonstrations and
tests need a source of plausible data. Second, threshold calibration requires
samples whose contamination is known by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudyDesign:
    """Parameters of a simulated two-arm study."""

    n_per_arm: int = 40
    n_items: int = 12
    scale_points: int = 5
    arm_difference: float = 0.0
    respondent_sd: float = 0.6
    item_sd: float = 0.5
    centre: float = 3.3


ARCHETYPES = ("invariant", "random", "satisficer", "duplicate", "hurried_genuine")


def _ordinal(latent: np.ndarray, points: int) -> np.ndarray:
    return np.clip(np.rint(latent), 1, points).astype(int)


def clean_sample(design: StudyDesign, seed: int = 0) -> pd.DataFrame:
    """Attentive respondents only."""
    rng = np.random.default_rng(seed)
    frames = []
    for index, arm in enumerate(("treatment", "control")):
        shift = design.arm_difference if arm == "treatment" else 0.0
        person = rng.normal(0, design.respondent_sd, design.n_per_arm)[:, None]
        noise = rng.normal(0, design.item_sd, (design.n_per_arm, design.n_items))
        latent = design.centre + shift + person + noise
        block = pd.DataFrame(
            _ordinal(latent, design.scale_points),
            columns=[f"item_{i + 1:02d}" for i in range(design.n_items)],
        )
        block.insert(0, "arm", arm)
        block.insert(1, "respondent_id", range(index * design.n_per_arm, (index + 1) * design.n_per_arm))
        block["duration_s"] = rng.lognormal(5.2, 0.7, design.n_per_arm).round()
        block["origin"] = rng.choice([f"origin_{i:02d}" for i in range(12)], design.n_per_arm)
        block["contaminated"] = False
        block["archetype"] = "attentive"
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def contaminate(
    frame: pd.DataFrame, prevalence: float, archetype: str = "invariant", seed: int = 0
) -> pd.DataFrame:
    """Replace a share of records with low-effort responding of a given kind."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype: {archetype}")

    rng = np.random.default_rng(seed)
    out = frame.copy()
    items = [c for c in out.columns if c.startswith("item_")]
    points = int(out[items].to_numpy().max())
    n_affected = int(round(prevalence * len(out)))
    if n_affected == 0:
        return out
    targets = rng.choice(out.index, n_affected, replace=False)

    for index in targets:
        if archetype == "invariant":
            out.loc[index, items] = rng.integers(1, points + 1)
            out.loc[index, "duration_s"] = rng.integers(20, 60)
        elif archetype == "random":
            out.loc[index, items] = rng.integers(1, points + 1, len(items))
            out.loc[index, "duration_s"] = rng.integers(25, 80)
        elif archetype == "satisficer":
            half = len(items) // 2
            out.loc[index, items[half:]] = out.loc[index, items[half]]
            out.loc[index, "duration_s"] = rng.integers(40, 120)
        elif archetype == "duplicate":
            donor = rng.choice(out.index)
            out.loc[index, items] = out.loc[donor, items].to_numpy()
            out.loc[index, "origin"] = out.loc[donor, "origin"]
            out.loc[index, "duration_s"] = rng.integers(20, 70)
        elif archetype == "hurried_genuine":
            out.loc[index, "duration_s"] = rng.integers(30, 70)

        out.loc[index, "contaminated"] = archetype != "hurried_genuine"
        out.loc[index, "archetype"] = archetype

    return out


def demonstration_study(seed: int = 0) -> pd.DataFrame:
    """Ready-made sample used by the examples and the test suite."""
    design = StudyDesign(n_per_arm=45, arm_difference=0.15)
    sample = clean_sample(design, seed=seed)
    sample = contaminate(sample, prevalence=0.12, archetype="invariant", seed=seed + 1)
    sample = contaminate(sample, prevalence=0.06, archetype="duplicate", seed=seed + 2)
    sample["record_key"] = sample["arm"] + ":" + sample["respondent_id"].astype(str)
    return sample
