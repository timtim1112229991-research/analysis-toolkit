# -*- coding: utf-8 -*-
"""Re-estimate the between-arm contrast without assuming independent records.

Primary specification is the cluster bootstrap over recruiting origins. The
matched pair analysis over one to one origins is reported alongside it as a
sensitivity check, not as a replacement.
"""

import os
import sys

import pandas as pd

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.clustering import (  # noqa: E402
    DEFAULT_SEED,
    cluster_equivalence,
    cluster_robust_linear,
    diagnose,
    paired_contrast,
    summarise_fit,
)
from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

ARMS = ('multi_agent', 'single_model')
MARGIN = 0.4
OUT = outputs('clustered')

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema)

items = [item for members in schema.dimensions.values() for item in members]
dimensions = list(schema.dimensions.keys())
frame['overall'] = frame[items].mean(axis=1)
targets = items + dimensions + ['overall']

print('=' * 96)
print('CLUSTER DIAGNOSIS')
print('=' * 96)
report = diagnose(frame, 'arm', 'origin')
print(report.as_table().to_string())
report.as_table().to_csv(os.path.join(OUT, 'cluster_diagnosis.csv'))

print()
print('=' * 96)
print('EQUIVALENCE UNDER A CLUSTER BOOTSTRAP, MARGIN %.2f' % MARGIN)
print('=' * 96)
rows = []
for target in targets:
    result = cluster_equivalence(frame, target, 'arm', 'origin', ARMS,
                                 margin=MARGIN, n_boot=4000)
    result['measure'] = target
    rows.append(result)
clustered = pd.DataFrame(rows).set_index('measure')
print(clustered[['difference', 'cluster_se', 'lower', 'upper', 'verdict', 'naive_p']].to_string())
clustered.to_csv(os.path.join(OUT, 'equivalence_clustered.csv'))

print()
print('=' * 96)
print('MATCHED PAIR SENSITIVITY OVER ONE TO ONE ORIGINS')
print('=' * 96)
rows = []
for target in targets:
    result = paired_contrast(frame, target, 'arm', 'origin', ARMS, margin=MARGIN)
    result['measure'] = target
    rows.append(result)
paired = pd.DataFrame(rows).set_index('measure')
print(paired.to_string())
paired.to_csv(os.path.join(OUT, 'equivalence_paired.csv'))

print()
print('=' * 96)
print('ADOPTION MODEL WITH ORIGIN CLUSTERED STANDARD ERRORS')
print('=' * 96)
frame['multi_agent'] = (frame['arm'] == 'multi_agent').astype(int)
predictors = ['logic', 'collaboration', 'accuracy', 'implementability',
              'transparency', 'controllability', 'multi_agent']
fit = cluster_robust_linear(frame, 'intention', predictors, 'origin')
table = summarise_fit(fit)
print(table.to_string())
print()
print('clusters used: %d | observations: %d | r squared: %.3f'
      % (fit.df_resid and frame['origin'].nunique(), int(fit.nobs), fit.rsquared))
table.to_csv(os.path.join(OUT, 'adoption_cluster_robust.csv'))

print()
print('=' * 96)
print('COMPARISON OF VERDICTS')
print('=' * 96)
summary = pd.DataFrame({
    'clustered': clustered['verdict'],
    'paired': paired['verdict'],
})
print(summary.to_string())
summary.to_csv(os.path.join(OUT, 'verdict_comparison.csv'))

record_run('run_clustered', {
    'margin': MARGIN,
    'margin_unit': 'scale points',
    'alpha': 0.05,
    'arms': list(ARMS),
    'cluster': 'origin',
    'bootstrap_seed': DEFAULT_SEED,
    'records': int(len(frame)),
    'clusters': int(frame['origin'].nunique()),
    'schema_digest': schema_digest(),
})

print()
print('outputs written to:', OUT)
