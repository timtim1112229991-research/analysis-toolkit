# -*- coding: utf-8 -*-
"""Re-estimate everything that still assumed independent records.

Three quantities are affected. The proportional odds model, whose odds ratios
are quoted in the abstract. The screening severity sweep, which underwrites the
claim that the one reversal is a power artefact. And the resolution limit, which
the discussion uses to explain why some contrasts are inconclusive.
"""

import os
import sys

import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.clustering import (  # noqa: E402
    DEFAULT_SEED,
    cluster_bootstrap_coefficients,
    cluster_equivalence,
)
from src.integrity import Thresholds, flags, signals  # noqa: E402
from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

ARMS = ('multi_agent', 'single_model')
MARGIN = 0.4
OUT = outputs('clustered')

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema).reset_index(drop=True)

items = [i for members in schema.dimensions.values() for i in members]
dimensions = list(schema.dimensions.keys())
frame['overall'] = frame[items].mean(axis=1)
frame['multi_agent'] = (frame['arm'] == ARMS[0]).astype(int)

PREDICTORS = ['logic', 'collaboration', 'accuracy', 'implementability',
              'transparency', 'controllability', 'multi_agent']


def ordinal_params(sample: pd.DataFrame) -> pd.Series:
    y = sample['intention'].astype(int)
    x = sample[PREDICTORS].astype(float)
    fit = OrderedModel(y, x, distr='logit').fit(method='bfgs', disp=False)
    return fit.params[PREDICTORS]


print('=' * 100)
print('1. PROPORTIONAL ODDS MODEL WITH CLUSTER BOOTSTRAP INTERVALS')
print('=' * 100)
naive = OrderedModel(frame['intention'].astype(int), frame[PREDICTORS].astype(float),
                     distr='logit').fit(method='bfgs', disp=False)
table = cluster_bootstrap_coefficients(frame, 'origin', ordinal_params,
                                       n_boot=1200, exponentiate=True)
table['naive_se'] = naive.bse[PREDICTORS].round(4)
table['naive_p'] = naive.pvalues[PREDICTORS].round(4)
table['se_inflation'] = (table['cluster_se'] / table['naive_se']).round(2)
print(table[['coefficient', 'naive_se', 'cluster_se', 'se_inflation',
             'or_coefficient', 'or_lower', 'or_upper', 'excludes_zero', 'naive_p']].to_string())
print()
print('resamples used: %d | failed fits: %d | clusters: %d'
      % (table.attrs['resamples'], table.attrs['failures'], table.attrs['clusters']))
print()
print('Predictors whose clustered interval excludes zero:')
print('  ' + ', '.join(table[table['excludes_zero']].index.tolist()))
table.to_csv(os.path.join(OUT, 'adoption_ordinal_clustered.csv'))

print()
print('=' * 100)
print('2. SCREENING SEVERITY SWEEP UNDER CLUSTERED INFERENCE')
print('=' * 100)
signal_frame = signals(frame, items, open_text=['open_positive', 'open_negative', 'open_process'])
flag_frame = flags(signal_frame, Thresholds(), items_count=len(items))
frame['flag_count'] = flag_frame.sum(axis=1).to_numpy()

sweep = []
for severity in range(0, 6):
    retained = frame[frame['flag_count'] < severity] if severity else frame
    if severity == 0:
        retained = frame
    if len(retained) < 12 or retained['arm'].nunique() < 2:
        sweep.append({'severity': severity, 'retained': len(retained),
                      'clusters': retained['origin'].nunique(), 'note': 'sample exhausted'})
        continue
    row = {'severity': severity, 'retained': len(retained),
           'clusters': int(retained['origin'].nunique())}
    for measure in ['overall'] + dimensions:
        result = cluster_equivalence(retained, measure, 'arm', 'origin', ARMS,
                                     margin=MARGIN, n_boot=1500)
        row[measure] = result['verdict']
        row[f'{measure}_diff'] = result['difference']
    sweep.append(row)

sweep_table = pd.DataFrame(sweep).set_index('severity')
print(sweep_table.to_string())
sweep_table.to_csv(os.path.join(OUT, 'sensitivity_clustered.csv'))
print()
print('Severity is the number of integrity flags at which a record is removed.')
print('Level 0 retains everything. A stable row of verdicts means the conclusion')
print('does not depend on where the screening line is drawn.')

print()
print('=' * 100)
print('3. RESOLUTION UNDER CLUSTERED INFERENCE')
print('=' * 100)
rows = []
for measure in items + dimensions + ['overall']:
    result = cluster_equivalence(frame, measure, 'arm', 'origin', ARMS,
                                 margin=MARGIN, n_boot=2500)
    half_width = (result['upper'] - result['lower']) / 2
    rows.append({'measure': measure,
                 'difference': result['difference'],
                 'half_width': round(half_width, 4),
                 'margin': MARGIN,
                 'resolvable': half_width < MARGIN,
                 'verdict': result['verdict']})
resolution = pd.DataFrame(rows).set_index('measure')
print(resolution.to_string())
print()
print('half width below the margin: %d of %d measures'
      % (int(resolution['resolvable'].sum()), len(resolution)))
print('median half width: %.3f | margin: %.2f'
      % (resolution['half_width'].median(), MARGIN))
resolution.to_csv(os.path.join(OUT, 'resolution_clustered.csv'))

record_run('reestimate', {
    'margin': MARGIN,
    'margin_unit': 'scale points',
    'arms': list(ARMS),
    'cluster': 'origin',
    'bootstrap_seed': DEFAULT_SEED,
    'resamples_adoption_model': 1200,
    'resamples_screening_sweep': 1500,
    'resamples_resolution_limit': 2500,
    'records': int(len(frame)),
    'clusters': int(frame['origin'].nunique()),
    'schema_digest': schema_digest(),
})

print()
print('outputs written to:', OUT)
