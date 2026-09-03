# -*- coding: utf-8 -*-
"""How often is each predictor positive across cluster resamples?

A percentile interval on a logit coefficient has heavy tails at this sample
size, so an interval that includes zero can conceal a resample distribution
that is almost entirely on one side. The share of resamples above zero states
that directly and is the fairer summary for coherence.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema).reset_index(drop=True)
frame['multi_agent'] = (frame['arm'] == 'multi_agent').astype(int)

PREDICTORS = ['logic', 'collaboration', 'accuracy', 'implementability',
              'transparency', 'controllability', 'multi_agent']
N_BOOT = 1500

keys = frame['origin'].unique()
blocks = {k: b for k, b in frame.groupby('origin')}
rng = np.random.default_rng(20260902)

draws, failures = [], 0
for _ in range(N_BOOT):
    chosen = rng.choice(keys, size=len(keys), replace=True)
    sample = pd.concat([blocks[k] for k in chosen], ignore_index=True)
    try:
        fit = OrderedModel(sample['intention'].astype(int),
                           sample[PREDICTORS].astype(float), distr='logit'
                           ).fit(method='bfgs', disp=False)
        draws.append(fit.params[PREDICTORS])
    except Exception:
        failures += 1

matrix = pd.DataFrame(draws)
observed = OrderedModel(frame['intention'].astype(int), frame[PREDICTORS].astype(float),
                        distr='logit').fit(method='bfgs', disp=False).params[PREDICTORS]

summary = pd.DataFrame({
    'coefficient': observed.round(4),
    'odds_ratio': np.exp(observed).round(2),
    'share_positive': (matrix > 0).mean().round(4),
    'median_resample_or': np.exp(matrix.median()).round(2),
    'p10_or': np.exp(matrix.quantile(0.10)).round(2),
    'p90_or': np.exp(matrix.quantile(0.90)).round(2),
})
summary['two_sided_equivalent'] = (2 * np.minimum(summary['share_positive'],
                                                  1 - summary['share_positive'])).round(4)

print('resamples used: %d | failures: %d | clusters: %d' % (len(draws), failures, len(keys)))
print()
print(summary.sort_values('share_positive', ascending=False).to_string())
print()
print('The share positive is the proportion of cluster resamples in which the')
print('predictor pushed intention upward. The final column converts it to a')
print('two sided probability, which is comparable with a conventional test.')

OUT = outputs('clustered')
summary.to_csv(os.path.join(OUT, 'coefficient_resample_shares.csv'))

record_run('coherence_share', {
    'predictors': PREDICTORS,
    'cluster': 'origin',
    'resampling_seed': 20260902,
    'resamples_requested': N_BOOT,
    'records': int(len(frame)),
    'clusters': int(frame['origin'].nunique()),
    'schema_digest': schema_digest(),
})

print()
print('written to:', os.path.join(OUT, 'coefficient_resample_shares.csv'))
