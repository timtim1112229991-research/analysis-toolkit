# Two-Arm Ordinal Instrument Analysis Toolkit

Analysis code for controlled comparisons that use short ordinal instruments administered to
practitioners. The toolkit was written for evaluations where two system configurations are compared
on the same task and the outcome is a professional judgement rather than an automatic score.

It covers four things that such studies commonly get wrong: treating a non-significant difference as
evidence of similarity, ignoring multiplicity across instrument items, screening records without
declaring the rules, and reporting a single estimate rather than the family of estimates that
different screening choices produce.

## Contents

| Module | Responsibility |
|---|---|
| `src/schema.py` | Positional description of source files, generated locally and never committed |
| `src/loading.py` | Assembly of a tidy analysis frame, composites and a data dictionary |
| `src/reliability.py` | Coefficient alpha with bootstrap intervals, item diagnostics |
| `src/comparison.py` | Rank-based and Welch contrasts, effect sizes, false discovery rate control |
| `src/equivalence.py` | Two one-sided tests against a declared margin, smallest detectable difference |
| `src/integrity.py` | Response integrity signals, declared thresholds, record-level audit log |
| `src/sensitivity.py` | Progressive exclusion and conclusion stability across screening severity |
| `src/adoption.py` | Proportional odds and least squares models for the adoption outcome |
| `src/qualitative.py` | Language-neutral substance descriptors and coder agreement |
| `src/coding.py` | Blinded coding sheets, chance-corrected agreement, adjudication and theme frequencies |
| `src/reporting.py` | Output writing with a provenance manifest for every run |
| `src/simulate.py` | Synthetic samples with contamination of known type and prevalence |
| `src/run_analysis.py` | Command line pipeline tying the above together |

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate      # on Windows
pip install -r requirements.txt
```

## Running the demonstration

No records are distributed with this repository. The pipeline can be exercised end to end on
generated data:

```bash
python -m src.run_analysis --synthetic --outputs outputs
```

Results are written to `outputs/`, which is excluded from version control.

## Running on real records

Source workbooks are described by column position, never by header text, so that no study-specific
or source-language content enters the code base. Generate a schema template, complete it, and keep
it local:

```bash
python -m examples.build_schema --workbook path/to/file.xlsx --items 12 22 --out config/study.schema.json
python -m src.run_analysis --schema config/study.schema.json --input "path/to/*.xlsx" \
    --margin 0.40 --margin-basis "half of one scale point" --declared-on 2026-09-02
```

The `config/` directory is excluded from version control because a completed schema echoes labels
from the source instrument.

## Declaring an equivalence margin

The margin must be fixed before the equivalence tests are computed, and the justification is
recorded in the run manifest alongside the results. A margin chosen after inspecting the data
invalidates the test, and nothing in this code can detect that, so the discipline is procedural.

## Screening discipline

Integrity thresholds are supplied explicitly rather than inherited from convention. Flag computation
is deliberately separated from outcome analysis so that screening can be applied while blinded to
results. Every record-level decision is written to an audit log, and the pipeline always reports the
whole sensitivity family rather than a single preferred estimate.

## Release policy

This repository contains source code only. Input data, derived data, results tables, figures, model
artefacts and logs are excluded by `.gitignore` and must not be committed. Verify with
`git status --porcelain` before every push.

## Tests

```bash
pip install pytest
pytest -q
```
