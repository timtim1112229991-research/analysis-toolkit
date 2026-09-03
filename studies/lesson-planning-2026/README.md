# Lesson planning study, 2026

The analysis behind a two-arm comparison of an AI lesson planning tool, in which
teachers used either a multi-agent configuration or a single model and then
rated the artefact and the experience on a twelve item ordinal instrument.

Everything here is the code and the provenance of that analysis. The response
records are held in restricted storage and are not part of this repository, so
a clone will not reproduce the numbers on its own. What a clone does give you is
the exact procedure, the exact configuration it read the records through, and
the exact parameters behind each run, which is what is needed to audit a
reported estimate or to apply the same procedure to your own records.

## What is here

| File | What it is |
|---|---|
| `study.schema.json` | The column map. Which position in the source workbooks holds which item, how the five ordinal labels are ordered, which items form each dimension, and which workbook belongs to which arm |
| `record_snapshot.json` | The freeze. A digest of each source workbook, a digest of the harmonised analysable table, and the counts by arm and wave. Re-running `_freeze_records.py --verify` reports whether the records still match it |
| `manifests/` | One provenance record per script. Each states the commit, the interpreter, the platform and the parameter values behind that run |
| `_paths.py` | Where the scripts read from and write to |
| the remaining scripts | The drivers, described below |

## The drivers

The estimators live in `src/` at the repository root. The scripts here are thin
drivers: they load the records through the schema, call an estimator, and write
a result. No statistical procedure is implemented in this directory.

| Script | What it produces |
|---|---|
| `_freeze_records.py` | Writes or verifies `record_snapshot.json` |
| `_run_clustered.py` | The primary between-arm contrast under a cluster bootstrap, with the matched pair analysis alongside it |
| `_reestimate.py` | The adoption model with cluster bootstrap intervals, the screening severity sweep, and the resolution limit |
| `_spec_checks.py` | Whether the adoption result survives a different predictor set, a split by wave, the removal of any one origin, and correlated predictors |
| `_coherence_share.py` | The share of cluster resamples in which each predictor pushed intention upward |
| `_check_pairing.py` | Whether a one to one origin holds one person seen twice or two colleagues |
| `_check_sessions.py` | The session timeline within confirmed pairs, and whether fill windows overlap |
| `_check_order.py` | Which arm a paired respondent saw first, and the paired contrast on unambiguous pairs |
| `_check_carryover.py` | Whether the equivalence finding survives removing repeat submissions |
| `_check_crossover.py` | Whether the two arms drew on independent respondents |
| `_coding_frames.py` | The coding frames for the three open response fields |
| `_build_coding_sheets.py` | Blinded coding sheets and codebooks for two independent coders |
| `_dump_open_text.py` | Prints the open responses for coding frame development. Output stays local |
| `_check_reproducibility.py` | Refits the adoption model and compares it with the stored result |

## Running it

Install the requirements at the repository root, then point the two environment
variables at your own records and a writable directory.

```bash
pip install -r requirements.txt

export STUDY_DATA=/path/to/workbooks
export STUDY_OUTPUTS=/path/to/results
```

The main pipeline is invoked from the repository root. The reported run used:

```bash
python -m src.run_analysis \
  --schema studies/lesson-planning-2026/study.schema.json \
  --input '/path/to/workbooks/*.xlsx' \
  --outputs /path/to/results \
  --margin 0.40 \
  --margin-basis 'Two fifths of one scale point. Fixed after collection and used as a sensitivity benchmark, not as a pre-specified threshold.' \
  --declared-on '2026-09-02, after collection' \
  --alpha 0.05 \
  --seed 20260902 \
  --outcome intention \
  --predictors logic collaboration accuracy implementability transparency controllability
```

The drivers are then run from this directory, in this order, since the later
ones read the record freeze written by the first:

```bash
cd studies/lesson-planning-2026
python _freeze_records.py
python _run_clustered.py
python _reestimate.py
python _spec_checks.py
python _coherence_share.py
python _check_pairing.py
python _check_sessions.py
python _check_order.py
python _check_carryover.py
python _check_crossover.py
```

Each writes its manifest into `manifests/` as it finishes.

## Determinism

Every resampling procedure is seeded. The seed is `DEFAULT_SEED` in
`src/clustering.py`, and the drivers do not override it, so the intervals are
reproducible rather than merely similar. On the reported records the adoption
model returns the same odds ratios and the same interval bounds on every run.

## The margin

The equivalence margin of 0.4 scale points was fixed after collection. It is
used as a sensitivity benchmark, not as a pre-specified threshold, and the
manifests record that wording alongside the value so the distinction cannot be
lost. Conclusions that depend on the margin are reported across a range of
screening severities rather than at one setting.
