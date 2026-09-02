"""Command line entry point for the two-arm evaluation pipeline.

Usage with real records requires a locally generated schema that is never
committed. Passing --synthetic runs the same pipeline on generated data, which
is how the examples and the test suite exercise the code.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from . import adoption, comparison, integrity, qualitative, reliability, reporting, sensitivity, simulate
from .equivalence import Margin, equivalence_table, smallest_detectable_difference
from .loading import data_dictionary, load_study
from .schema import Schema


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-arm ordinal instrument analysis")
    parser.add_argument("--schema", help="path to a locally generated schema description")
    parser.add_argument("--input", help="glob pattern matching the source workbooks")
    parser.add_argument("--outputs", default="outputs", help="directory for results, excluded from git")
    parser.add_argument("--synthetic", action="store_true", help="run on generated data instead")
    parser.add_argument("--margin", type=float, default=0.40, help="equivalence margin on the raw scale")
    parser.add_argument("--margin-basis", default="", help="justification recorded with the margin")
    parser.add_argument("--declared-on", default="", help="date the margin was fixed")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--outcome", help="item modelled as the adoption outcome")
    parser.add_argument("--predictors", nargs="*", help="items entered as predictors of the outcome")
    return parser.parse_args(argv)


def _synthetic_inputs() -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str]]:
    frame = simulate.demonstration_study()
    items = [c for c in frame.columns if c.startswith("item_")]
    dimensions = {
        "dimension_a": items[:5],
        "dimension_b": items[5:9],
        "dimension_c": items[9:],
    }
    for name, members in dimensions.items():
        frame[name] = frame[members].mean(axis=1)
    frame["overall"] = frame[items].mean(axis=1)
    frame["sum_score"] = frame[items].sum(axis=1)
    frame["wave"] = "single"
    return frame, items, dimensions, []


def _real_inputs(arguments: argparse.Namespace):
    if not arguments.schema or not arguments.input:
        raise SystemExit("--schema and --input are required unless --synthetic is given")
    schema = Schema.from_json(arguments.schema)
    paths = sorted(glob.glob(arguments.input))
    if not paths:
        raise SystemExit(f"no files matched {arguments.input}")
    frame = load_study(paths, schema)
    return frame, schema.item_names(), schema.dimensions, list(schema.columns.open_text.keys()), schema


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    destination = reporting.output_directory(arguments.outputs)
    schema = None

    if arguments.synthetic:
        frame, items, dimensions, open_fields = _synthetic_inputs()
    else:
        frame, items, dimensions, open_fields, schema = _real_inputs(arguments)

    measures = items + list(dimensions) + ["overall", "sum_score"]
    margin = Margin(
        value=arguments.margin,
        unit="scale points",
        justification=arguments.margin_basis or "not recorded",
        declared_on=arguments.declared_on or "not recorded",
    )

    signal_frame = integrity.signals(frame, items, open_fields or None)
    thresholds = integrity.Thresholds()
    flag_frame = integrity.flags(signal_frame, thresholds, len(items))

    sections: dict[str, pd.DataFrame] = {}

    sections["Sample by arm and wave"] = pd.crosstab(frame["arm"], frame["wave"], margins=True)
    sections["Scale reliability"] = pd.DataFrame([reliability.alpha_with_interval(frame[items])])
    sections["Dimension reliability"] = reliability.dimension_summary(frame, dimensions)
    sections["Item diagnostics"] = reliability.item_diagnostics(frame[items])

    contrasts = comparison.compare(frame, measures, seed=arguments.seed)
    sections["Arm comparison"] = contrasts
    sections["Top box percentages"] = comparison.top_box(frame, items, threshold=4)

    equivalence = equivalence_table(frame, measures, margin, alpha=arguments.alpha)
    sections["Equivalence tests"] = equivalence
    sections["Smallest detectable difference"] = pd.DataFrame(
        {"measure": list(dimensions) + ["overall"],
         "value": [smallest_detectable_difference(frame, m) for m in list(dimensions) + ["overall"]]}
    ).set_index("measure")

    sections["Integrity signal prevalence"] = integrity.summarise(flag_frame)
    family = sensitivity.progressive_exclusion(
        frame, flag_frame, list(dimensions) + ["overall"], margin=margin
    )
    sections["Sensitivity family"] = family
    sections["Conclusion stability"] = sensitivity.conclusion_stability(family)

    outcome = arguments.outcome or items[-1]
    if outcome not in items:
        raise SystemExit(f"outcome {outcome} is not an instrument item")
    predictors = arguments.predictors or [i for i in items if i != outcome][:6]
    predictors = [p for p in predictors if p != outcome]
    try:
        sections["Adoption model, ordinal"] = adoption.ordinal_model(frame, outcome, predictors)
    except Exception as error:  # model failure must not discard the rest of the run
        sections["Adoption model, ordinal"] = pd.DataFrame({"error": [str(error)]})
    sections["Adoption model, linear"] = adoption.linear_model(frame, outcome, predictors)

    if open_fields:
        sections["Open response substance"] = qualitative.substance(frame, open_fields)
        sections["Repeated open responses"] = qualitative.duplicate_responses(frame, open_fields)

    if schema is not None:
        sections["Data dictionary"] = data_dictionary(frame, schema)

    for title, table in sections.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            reporting.write_table(table, destination, title.lower().replace(" ", "_").replace(",", ""))

    reporting.write_table(
        integrity.audit_log(frame, flag_frame), destination, "integrity_audit_log", index=False
    )
    reporting.write_console_report(sections, destination, "analysis_report")
    manifest = reporting.run_manifest(
        destination,
        {
            "synthetic": arguments.synthetic,
            "margin": margin.as_row(),
            "alpha": arguments.alpha,
            "seed": arguments.seed,
            "items": len(items),
            "records": int(len(frame)),
        },
    )

    print(f"records analysed: {len(frame)}")
    print(f"outputs written to: {Path(arguments.outputs).resolve()}")
    print(f"run commit: {manifest['commit']}")


if __name__ == "__main__":
    main()
