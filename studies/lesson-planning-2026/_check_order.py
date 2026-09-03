# -*- coding: utf-8 -*-
"""Consequences of the pairing evidence.

Three questions follow from establishing that most one to one origins hold a
single respondent seen twice. Which arm did that respondent see first, how many
distinct people the study actually recruited, and whether the paired contrast
survives when restricted to the pairs whose identity evidence is unambiguous.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.comparison import benjamini_hochberg  # noqa: E402
from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

STABLE = ['experience', 'stage', 'subject']
ARMS = ('multi_agent', 'single_model')
MARGIN = 0.4

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema).reset_index(drop=True)

items = [i for members in schema.dimensions.values() for i in members]
dimensions = list(schema.dimensions.keys())
frame['overall'] = frame[items].mean(axis=1)
targets = items + dimensions + ['overall']

counts = frame.groupby(['origin', 'arm']).size().unstack(fill_value=0)
one_to_one = counts[(counts.get(ARMS[0], 0) == 1) & (counts.get(ARMS[1], 0) == 1)].index

pairs = []
for origin in one_to_one:
    block = frame[frame['origin'] == origin]
    left = block[block['arm'] == ARMS[0]].iloc[0]
    right = block[block['arm'] == ARMS[1]].iloc[0]
    concordance = sum(1 for f in STABLE if str(left[f]) == str(right[f]))
    gap = abs((left['submitted_at'] - right['submitted_at']).total_seconds()) / 3600
    pairs.append({'origin': origin, 'left': left, 'right': right,
                  'concordance': concordance, 'gap': gap,
                  'confirmed': concordance == 3 and gap < 24})

confirmed = [p for p in pairs if p['confirmed']]

print('=' * 96)
print('1. ORDER OF EXPOSURE WITHIN CONFIRMED PAIRS')
print('=' * 96)
first = ['multi_agent' if p['left']['submitted_at'] < p['right']['submitted_at']
         else 'single_model' for p in confirmed]
tally = pd.Series(first).value_counts()
print(tally.to_string())
binomial = stats.binomtest(int(tally.get('multi_agent', 0)), len(confirmed), 0.5)
print('\nbalance of first exposure, exact binomial p = %.4f' % binomial.pvalue)
print('interpretation: an unbalanced order would confound arm with position')

print()
print('=' * 96)
print('2. HOW MANY PEOPLE THE STUDY ACTUALLY RECRUITED')
print('=' * 96)
per_origin = frame.groupby(['origin', 'arm']).size().unstack(fill_value=0)
for arm in ARMS:
    if arm not in per_origin:
        per_origin[arm] = 0
both = per_origin[list(ARMS)].min(axis=1)
distinct = (per_origin[list(ARMS)].max(axis=1)).sum()
print('records:                                    %3d' % len(frame))
print('origins:                                    %3d' % len(per_origin))
print('records attributable to a repeat visit:     %3d' % int(both.sum()))
print('lower bound on distinct respondents:        %3d' % int(distinct))
print('effective sample if every crossing repeats: %3d' % int(distinct))
print('\nassumption: at an origin serving both arms, the smaller count is people')
print('seen twice and the surplus is additional colleagues')

print()
print('=' * 96)
print('3. PAIRED CONTRAST ON CONFIRMED PAIRS ONLY (n = %d)' % len(confirmed))
print('=' * 96)
rows = []
for target in targets:
    left = np.array([p['left'][target] for p in confirmed], dtype=float)
    right = np.array([p['right'][target] for p in confirmed], dtype=float)
    difference = left - right
    n = len(difference)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(n))
    critical = stats.t.ppf(0.95, df=n - 1)
    lower, upper = mean - critical * se, mean + critical * se
    signed = (float('nan') if len(set(difference)) == 1
              else float(stats.wilcoxon(left, right, zero_method='zsplit').pvalue))
    rows.append({'measure': target, 'mean_difference': round(mean, 4),
                 'se': round(se, 4), 'lower': round(lower, 4), 'upper': round(upper, 4),
                 'wilcoxon_p': round(signed, 4),
                 'verdict': 'equivalent' if (lower > -MARGIN and upper < MARGIN)
                 else 'inconclusive'})

table = pd.DataFrame(rows).set_index('measure')
table['adjusted_p'] = benjamini_hochberg(table['wilcoxon_p'].values).round(4)
print(table.to_string())

OUT = outputs('clustered')
table.to_csv(os.path.join(OUT, 'paired_confirmed.csv'))

record_run('check_order', {
    'margin': MARGIN,
    'margin_unit': 'scale points',
    'arms': list(ARMS),
    'cluster': 'origin',
    'concordance_fields': STABLE,
    'multiplicity_control': 'Benjamini and Hochberg, alpha 0.05',
    'records': int(len(frame)),
    'schema_digest': schema_digest(),
})

print()
print('=' * 96)
print('4. VERDICT COUNT')
print('=' * 96)
print(table['verdict'].value_counts().to_string())
print('\nmeasures surviving multiplicity control at 0.05:')
survivors = table[table['adjusted_p'] < 0.05]
print(survivors[['mean_difference', 'wilcoxon_p', 'adjusted_p']].to_string()
      if len(survivors) else '  none')
