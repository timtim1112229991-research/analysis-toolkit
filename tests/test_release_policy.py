"""Guards on what may enter the public repository.

These tests encode the release policy so that a violation fails the suite rather
than reaching a remote host.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_EXTENSIONS = {
    ".xlsx", ".xls", ".csv", ".tsv", ".parquet", ".sav", ".dta",
    ".docx", ".doc", ".pptx", ".pdf",
    ".png", ".jpg", ".jpeg", ".svg", ".eps", ".tif", ".tiff",
    ".pkl", ".joblib", ".h5", ".pt", ".onnx",
}

FORBIDDEN_DIRECTORIES = {
    "data", "raw", "input", "inputs", "outputs", "output", "results",
    "figures", "figs", "tables", "artefacts", "artifacts", "logs",
    "models", "checkpoints", "config",
}

FORBIDDEN_TERMS = ("cursoragent", "copilot generated")

NON_ASCII_ALLOWED = re.compile(r"[\u2013\u2018\u2019\u201c\u201d\u00a0]")
EM_DASH = "\u2014"


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip("not a git repository")
    return [ROOT / line for line in result.stdout.splitlines() if line]


def test_no_data_or_output_files_are_tracked():
    offenders = [f for f in tracked_files() if f.suffix.lower() in FORBIDDEN_EXTENSIONS
                 and not f.name.endswith(".example.json")]
    assert offenders == [], f"data or output files tracked: {offenders}"


def test_no_data_or_output_directories_are_tracked():
    offenders = [f for f in tracked_files()
                 if set(f.relative_to(ROOT).parts[:-1]) & FORBIDDEN_DIRECTORIES]
    assert offenders == [], f"restricted directories tracked: {offenders}"


def test_tracked_text_is_english_only():
    offenders = []
    for path in tracked_files():
        if path.suffix not in {".py", ".md", ".txt", ".json", ".yml", ".yaml", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        stripped = NON_ASCII_ALLOWED.sub("", text)
        if any(ord(ch) > 0x2100 for ch in stripped):
            offenders.append(path.name)
    assert offenders == [], f"non-English characters found in: {offenders}"


def test_no_em_dashes_in_tracked_text():
    offenders = []
    for path in tracked_files():
        if path.suffix not in {".py", ".md", ".txt"}:
            continue
        try:
            if EM_DASH in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        except (OSError, UnicodeDecodeError):
            continue
    assert offenders == [], f"em dash found in: {offenders}"


def test_no_automated_tool_is_credited():
    offenders = []
    for path in tracked_files():
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        if any(term in text for term in FORBIDDEN_TERMS):
            offenders.append(path.name)
    assert offenders == [], f"automated tool credited in: {offenders}"


def test_gitignore_covers_every_restricted_directory():
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [d for d in FORBIDDEN_DIRECTORIES if f"{d}/" not in patterns]
    assert missing == [], f"gitignore does not exclude: {missing}"
