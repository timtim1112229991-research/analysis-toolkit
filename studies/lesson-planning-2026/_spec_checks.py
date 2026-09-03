# -*- coding: utf-8 -*-
"""How fragile is the partnership effect?

The manuscript now rests on a single coefficient, so the checks below ask
whether it survives a different choice of predictors, a split by collection
wave, the removal of any one recruiting origin, and the presence of correlated
predictors that could be splitting a common signal.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from src.clustering import (  # noqa: E402
    DEFAULT_SEED,
    cluster_bootstrap_coefficients,
    cluster_robust_linear,
)
from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

OUT = outputs('clustered')

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema).reset_index(drop=True)
frame['multi_agent'] = (frame['arm'] == 'multi_agent').astype(int)

items = [i for m in schema.dimensions.values() for i in m]
OUTCOME = 'intention'
PRIMARY = ['logic', 'collaboration', 'accuracy', 'implementability',
           'transparency', 'controllability', 'multi_agent']
ALL_ITEMS = [i for i in items if i != OUTCOME] + ['multi_agent']


def ordinal_fit(sample, predictors):
    y = sample[OUTCOME].astype(int)
    x = sample[predictors].astype(float)
    return OrderedModel(y, x, distr='logit').fit(method='bfgs', disp=False)


def report(sample, predictors, label, n_boot=600):
    def estimator(s):
        return ordinal_fit(s, predictors).params[predictors]

    table = cluster_bootstrap_coefficients(sample, 'origin', estimator,
                                           n_boot=n_boot, exponentiate=True)
    row = table.loc['collaboration']
    return {
        'specification': label,
        'predictors': len(predictors),
        'records': len(sample),
        'origins': sample['origin'].nunique(),
        'odds_ratio': row['or_coefficient'],
        'lower': row['or_lower'],
        'upper': row['or_upper'],
        'excludes_one': bool(row['excludes_zero']),
    }


print('=' * 100)
print('1. ALTERNATIVE PREDICTOR SETS')
print('=' * 100)
sets = {
    'primary, six attributes plus arm': PRIMARY,
    'every item except the outcome, plus arm': ALL_ITEMS,
    'partnership and arm only': ['collaboration', 'multi_agent'],
    'partnership, coherence and arm': ['logic', 'collaboration', 'multi_agent'],
    'artefact attributes plus arm': ['logic', 'innovation', 'accuracy',
                                     'differentiation', 'implementability', 'multi_agent'],
    'interaction attributes plus arm': ['controllability', 'transparency',
                                        'cognitive_effort', 'collaboration', 'multi_agent'],
}
rows = []
for label, predictors in sets.items():
    if 'collaboration' not in predictors:
        continue
    rows.append(report(frame, predictors, label))
alt = pd.DataFrame(rows).set_index('specification')
print(alt.to_string())
alt.to_csv(os.path.join(OUT, 'spec_predictor_sets.csv'))

print()
print('=' * 100)
print('2. SPLIT BY COLLECTION WAVE')
print('=' * 100)
rows = []
for wave, block in frame.groupby('wave'):
    try:
        rows.append(report(block, PRIMARY, 'primary set, %s' % wave, n_boot=400))
    except Exception as error:
        rows.append({'specification': 'primary set, %s' % wave, 'records': len(block),
                     'origins': block['origin'].nunique(), 'odds_ratio': np.nan,
                     'lower': np.nan, 'upper': np.nan, 'excludes_one': False,
                     'predictors': len(PRIMARY)})
        print('  %s did not fit: %s' % (wave, error))
    try:
        rows.append(report(block, ['collaboration', 'multi_agent'],
                           'partnership only, %s' % wave, n_boot=400))
    except Exception as error:
        print('  %s reduced fit failed: %s' % (wave, error))
waves = pd.DataFrame(rows).set_index('specification')
print(waves.to_string())
waves.to_csv(os.path.join(OUT, 'spec_by_wave.csv'))

print()
print('=' * 100)
print('3. ARCHITECTURE BY WAVE INTERACTION')
print('=' * 100)
frame['wave_2'] = (frame['wave'] == 'wave_2').astype(int)
frame['arm_x_wave'] = frame['multi_agent'] * frame['wave_2']
interaction = cluster_robust_linear(
    frame, OUTCOME, ['collaboration', 'logic', 'multi_agent', 'wave_2', 'arm_x_wave'], 'origin')
table = pd.DataFrame({'coefficient': interaction.params.round(4),
                      'std_error': interaction.bse.round(4),
                      'p_value': interaction.pvalues.round(4)})
print(table.to_string())
print()
print('interaction probability: %.4f' % interaction.pvalues['arm_x_wave'])
print('an absent interaction means the two waves may be pooled')
table.to_csv(os.path.join(OUT, 'spec_interaction.csv'))

print()
print('=' * 100)
print('4. LEAVE ONE ORIGIN OUT, PRIMARY SPECIFICATION')
print('=' * 100)
base = ordinal_fit(frame, PRIMARY).params['collaboration']
rows = []
for origin in sorted(frame['origin'].unique()):
    reduced = frame[frame['origin'] != origin]
    try:
        fit = ordinal_fit(reduced, PRIMARY)
        rows.append({'origin_removed': origin,
                     'records_dropped': int((frame['origin'] == origin).sum()),
                     'coefficient': round(float(fit.params['collaboration']), 4),
                     'odds_ratio': round(float(np.exp(fit.params['collaboration'])), 3),
                     'p_value': round(float(fit.pvalues['collaboration']), 4)})
    except Exception:
        rows.append({'origin_removed': origin, 'records_dropped': int((frame['origin'] == origin).sum()),
                     'coefficient': np.nan, 'odds_ratio': np.nan, 'p_value': np.nan})

loo = pd.DataFrame(rows).set_index('origin_removed').sort_values('coefficient')
print('full sample coefficient: %.4f, odds ratio %.3f' % (base, np.exp(base)))
print()
print('five most influential removals, by lowest resulting coefficient:')
print(loo.head(5).to_string())
print()
print('five most influential removals, by highest resulting coefficient:')
print(loo.tail(5).to_string())
print()
print('coefficient range across all removals: %.4f to %.4f'
      % (loo['coefficient'].min(), loo['coefficient'].max()))
print('odds ratio range: %.2f to %.2f' % (loo['odds_ratio'].min(), loo['odds_ratio'].max()))
print('removals leaving the effect unconventional at 0.05: %d of %d'
      % (int((loo['p_value'] >= 0.05).sum()), len(loo)))
loo.to_csv(os.path.join(OUT, 'spec_leave_one_origin_out.csv'))

print()
print('=' * 100)
print('5. CORRELATION AND VARIANCE INFLATION AMONG PREDICTORS')
print('=' * 100)
attributes = [p for p in PRIMARY if p != 'multi_agent']
print('Spearman correlations:')
print(frame[attributes].corr(method='spearman').round(2).to_string())
print()
design = sm.add_constant(frame[attributes].astype(float))
vif = pd.DataFrame({
    'predictor': design.columns,
    'vif': [round(variance_inflation_factor(design.values, i), 2)
            for i in range(design.shape[1])],
}).set_index('predictor')
print(vif.to_string())
print()
print('Values above about five indicate that predictors are sharing signal,')
print('which would explain why coherence and partnership trade significance')
print('between specifications.')
vif.to_csv(os.path.join(OUT, 'spec_vif.csv'))

record_run('spec_checks', {
    'outcome': OUTCOME,
    'primary_predictors': PRIMARY,
    'cluster': 'origin',
    'bootstrap_seed': DEFAULT_SEED,
    'resamples_full_sample': 600,
    'resamples_within_wave': 400,
    'records': int(len(frame)),
    'clusters': int(frame['origin'].nunique()),
    'schema_digest': schema_digest(),
})

print()
print('outputs written to:', OUT)
