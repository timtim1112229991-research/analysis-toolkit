# -*- coding: utf-8 -*-
"""Reproduce a headline result from the public package and compare it.

The declarations claim the code that produced every reported estimate is
public. This tests that claim rather than repeating it: the adoption model is
refitted here using only the published toolkit, against the frozen records, and
the result is compared with the stored output the manuscript quotes.

Nothing is written. The stored outputs are left exactly as they are.
"""

import os
import sys

import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from _paths import OUTPUTS, SCHEMA, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.clustering import cluster_bootstrap_coefficients  # noqa: E402
from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

ARMS = ('multi_agent', 'single_model')
PREDICTORS = ['logic', 'collaboration', 'accuracy', 'implementability',
              'transparency', 'controllability', 'multi_agent']


def ordinal_params(sample):
    y = sample['intention'].astype(int)
    x = sample[PREDICTORS].astype(float)
    fit = OrderedModel(y, x, distr='logit').fit(method='bfgs', disp=False)
    return fit.params[PREDICTORS]


def main():
    schema = Schema.from_json(SCHEMA)
    paths = workbooks()
    frame = load_study(paths, schema).reset_index(drop=True)
    frame['multi_agent'] = (frame['arm'] == ARMS[0]).astype(int)

    print('Records loaded through the public loader: %d' % len(frame))
    print('Recruiting origins: %d' % frame['origin'].nunique())
    print()

    # No seed is passed, exactly as the original driver did, so this also
    # confirms the library default is what produced the reported intervals.
    fresh = cluster_bootstrap_coefficients(frame, 'origin', ordinal_params,
                                           n_boot=1200, exponentiate=True)
    stored = pd.read_csv(
        os.path.join(OUTPUTS, 'clustered', 'adoption_ordinal_clustered.csv')
    ).rename(columns={'Unnamed: 0': 'predictor'}).set_index('predictor')

    columns = ['or_coefficient', 'or_lower', 'or_upper', 'cluster_se']
    print('%-18s %-28s %-28s' % ('Predictor', 'Stored output', 'Recomputed now'))
    print('-' * 78)
    mismatches = 0
    for predictor in PREDICTORS:
        a = stored.loc[predictor, columns]
        b = fresh.loc[predictor, columns]
        agree = all(abs(float(a[c]) - float(b[c])) < 5e-4 for c in columns)
        if not agree:
            mismatches += 1
        print('%-18s %-28s %-28s %s'
              % (predictor,
                 '%.3f [%.3f, %.3f]' % (a['or_coefficient'], a['or_lower'], a['or_upper']),
                 '%.3f [%.3f, %.3f]' % (b['or_coefficient'], b['or_lower'], b['or_upper']),
                 'ok' if agree else 'MISMATCH'))
    print('-' * 78)
    print('resamples %d, failed fits %d, clusters %d'
          % (fresh.attrs['resamples'], fresh.attrs['failures'], fresh.attrs['clusters']))
    print('%d of %d predictors reproduce exactly'
          % (len(PREDICTORS) - mismatches, len(PREDICTORS)))


if __name__ == '__main__':
    main()
