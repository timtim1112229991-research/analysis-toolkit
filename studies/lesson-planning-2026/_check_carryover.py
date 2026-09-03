# -*- coding: utf-8 -*-
"""Is the equivalence finding partly an artefact of back to back rating?

Where one respondent completed both questionnaires in a single sitting, the
second set of ratings may be anchored on the first rather than formed from a
fresh evaluation. That would manufacture similarity between the arms. Three
checks bear on it: how far apart the two sets of ratings are within a pair
compared with unrelated respondents, and whether the equivalence conclusion
survives when the repeat submissions are removed.
"""

import os
import sys

import numpy as np
import pandas as pd

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.clustering import DEFAULT_SEED, cluster_equivalence  # noqa: E402
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

pairs, later_index = [], []
for origin in one_to_one:
    block = frame[frame['origin'] == origin]
    a = block[block['arm'] == ARMS[0]].iloc[0]
    b = block[block['arm'] == ARMS[1]].iloc[0]
    if sum(1 for f in STABLE if str(a[f]) == str(b[f])) != 3:
        continue
    pairs.append((a, b))
    later = a if a['submitted_at'] > b['submitted_at'] else b
    later_index.append(later.name)

print('=' * 96)
print('1. HOW SIMILAR ARE THE TWO RATING SETS WITHIN ONE RESPONDENT')
print('=' * 96)
within = np.array([np.abs(a[items].to_numpy(float) - b[items].to_numpy(float)).mean()
                   for a, b in pairs])

rng = np.random.default_rng(20260902)
left_pool = frame[frame['arm'] == ARMS[0]]
right_pool = frame[frame['arm'] == ARMS[1]]
between = []
while len(between) < 4000:
    a = left_pool.iloc[rng.integers(len(left_pool))]
    b = right_pool.iloc[rng.integers(len(right_pool))]
    if a['origin'] != b['origin']:
        between.append(np.abs(a[items].to_numpy(float) - b[items].to_numpy(float)).mean())
between = np.array(between)

print('mean absolute item difference')
print('  within a confirmed pair, same person:   %.3f  (n = %d)' % (within.mean(), len(within)))
print('  between unrelated respondents:          %.3f  (n = %d)' % (between.mean(), len(between)))
print('  ratio:                                  %.2f' % (within.mean() / between.mean()))
print()
print('permutation probability that pairs are no more alike than unrelated records: %.4f'
      % float((between <= within.mean()).mean()))
print()
print('A ratio far below one means the second questionnaire largely reproduces the')
print('first. That is expected of one person rating twice, and it is also what')
print('anchoring would produce, so this quantity cannot separate the two.')

print()
print('=' * 96)
print('2. EQUIVALENCE WITH REPEAT SUBMISSIONS REMOVED')
print('=' * 96)
reduced = frame.drop(index=later_index)
print('records retained: %d of %d | %s'
      % (len(reduced), len(frame), reduced['arm'].value_counts().to_dict()))
rows = []
for target in targets:
    result = cluster_equivalence(reduced, target, 'arm', 'origin', ARMS,
                                 margin=MARGIN, n_boot=3000)
    result['measure'] = target
    rows.append(result)
first_only = pd.DataFrame(rows).set_index('measure')
print(first_only[['difference', 'lower', 'upper', 'verdict']].to_string())

print()
print('=' * 96)
print('3. EQUIVALENCE ON ORIGINS SERVING ONE ARM ONLY')
print('=' * 96)
crossing = counts[(counts.get(ARMS[0], 0) > 0) & (counts.get(ARMS[1], 0) > 0)].index
clean = frame[~frame['origin'].isin(crossing)]
print('records retained: %d of %d | %s'
      % (len(clean), len(frame), clean['arm'].value_counts().to_dict()))
rows = []
for target in targets:
    try:
        result = cluster_equivalence(clean, target, 'arm', 'origin', ARMS,
                                     margin=MARGIN, n_boot=3000)
    except ValueError:
        continue
    result['measure'] = target
    rows.append(result)
clean_table = pd.DataFrame(rows).set_index('measure')
print(clean_table[['difference', 'lower', 'upper', 'verdict']].to_string())

print()
print('=' * 96)
print('4. VERDICTS SIDE BY SIDE')
print('=' * 96)
summary = pd.DataFrame({
    'all records': [cluster_equivalence(frame, t, 'arm', 'origin', ARMS,
                                        margin=MARGIN, n_boot=3000)['verdict'] for t in targets],
    'repeats removed': first_only['verdict'].reindex(targets).to_numpy(),
    'single arm origins': clean_table['verdict'].reindex(targets).to_numpy(),
}, index=targets)
print(summary.to_string())
print()
print(summary.apply(pd.Series.value_counts).fillna(0).astype(int).to_string())

OUT = outputs('clustered')
summary.to_csv(os.path.join(OUT, 'carryover_sensitivity.csv'))

record_run('check_carryover', {
    'margin': MARGIN,
    'margin_unit': 'scale points',
    'arms': list(ARMS),
    'cluster': 'origin',
    'bootstrap_seed': DEFAULT_SEED,
    'resamples': 3000,
    'concordance_fields': STABLE,
    'records': int(len(frame)),
    'clusters': int(frame['origin'].nunique()),
    'schema_digest': schema_digest(),
})

print()
print('written to:', os.path.join(OUT, 'carryover_sensitivity.csv'))
