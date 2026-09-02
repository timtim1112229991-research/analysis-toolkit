"""Generate a schema template from a workbook for local completion.

The template is written to a directory excluded from version control, because a
completed schema echoes response labels from the source instrument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schema import discover


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a schema template")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--items", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--out", default="config/study.schema.template.json")
    arguments = parser.parse_args()

    template = discover(arguments.workbook, arguments.items[0], arguments.items[1])
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"template written to {destination}")
    print("Complete the column indices, order the ordinal labels, then pass it to run_analysis.")


if __name__ == "__main__":
    main()
