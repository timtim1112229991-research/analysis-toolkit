"""Output writing and run provenance.

All outputs are written to a directory that is excluded from version control.
Each run records an identifier so that any reported number can be traced back
to the code state that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def output_directory(root: str | Path = "outputs") -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() or "not a repository"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def working_tree_clean() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() == ""
    except (OSError, subprocess.SubprocessError):
        return False


def run_manifest(destination: Path, parameters: dict[str, object]) -> dict[str, object]:
    """Write the provenance record that accompanies every set of results."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit_hash(),
        "working_tree_clean": working_tree_clean(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "parameters": parameters,
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_table(table: pd.DataFrame, destination: Path, name: str, index: bool = True) -> Path:
    path = destination / f"{name}.csv"
    table.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def write_console_report(sections: dict[str, pd.DataFrame], destination: Path, name: str) -> Path:
    """Single readable text summary alongside the machine-readable tables."""
    lines: list[str] = []
    for title, table in sections.items():
        lines.append("=" * 88)
        lines.append(title)
        lines.append("=" * 88)
        lines.append(table.to_string() if isinstance(table, pd.DataFrame) else str(table))
        lines.append("")
    path = destination / f"{name}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
