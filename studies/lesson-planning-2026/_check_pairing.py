# -*- coding: utf-8 -*-
"""Test whether a one to one origin holds one person or two colleagues.

Three sets of record pairs are compared on demographic concordance.

    A  cross arm pairs at an origin contributing exactly one record to each arm,
       the candidate same person pairs
    B  same arm pairs at a shared origin, who are necessarily different people
       and therefore give an empirical colleague baseline
    C  cross arm pairs drawn from different origins, giving an unrelated baseline

If A resembles B, the sixteen candidate pairs are colleagues on a shared network
and the matched pair analysis must be withdrawn. If A stands well clear of B,
the pairs are most plausibly one respondent seen twice.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

STABLE = ['experience', 'stage', 'subject']
ALL_FIELDS = STABLE + ['digital_confidence', 'tool_frequency']
ARMS = ('multi_agent', 'single_model')
SEED = 20260902

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema).reset_index(drop=True)


def matches(left, right, fields):
    return sum(1 for f in fields if str(left[f]) == str(right[f]))


counts = frame.groupby(['origin', 'arm']).size().unstack(fill_value=0)
one_to_one = counts[(counts.get(ARMS[0], 0) == 1) & (counts.get(ARMS[1], 0) == 1)].index

set_a, set_b, set_c = [], [], []

for origin in one_to_one:
    block = frame[frame['origin'] == origin]
    left = block[block['arm'] == ARMS[0]].iloc[0]
    right = block[block['arm'] == ARMS[1]].iloc[0]
    set_a.append((left, right))

for origin, block in frame.groupby('origin'):
    for arm in ARMS:
        same = block[block['arm'] == arm]
        for i, j in itertools.combinations(range(len(same)), 2):
            set_b.append((same.iloc[i], same.iloc[j]))

rng = np.random.default_rng(SEED)
left_pool = frame[frame['arm'] == ARMS[0]]
right_pool = frame[frame['arm'] == ARMS[1]]
while len(set_c) < 2000:
    left = left_pool.iloc[rng.integers(len(left_pool))]
    right = right_pool.iloc[rng.integers(len(right_pool))]
    if left['origin'] != right['origin']:
        set_c.append((left, right))


def summarise(pairs, name, fields):
    scores = np.array([matches(l, r, fields) for l, r in pairs], dtype=float)
    return {
        'set': name,
        'pairs': len(pairs),
        'mean_matches': round(float(scores.mean()), 3),
        'of_fields': len(fields),
        'all_match': int((scores == len(fields)).sum()),
        'all_match_pct': round(100 * float((scores == len(fields)).mean()), 1),
        'none_match': int((scores == 0).sum()),
    }


print('=' * 96)
print('DEMOGRAPHIC CONCORDANCE, THREE STABLE TRAITS')
print('=' * 96)
print(pd.DataFrame([
    summarise(set_a, 'A cross arm, one to one origin', STABLE),
    summarise(set_b, 'B same arm, shared origin', STABLE),
    summarise(set_c, 'C cross arm, different origin', STABLE),
]).to_string(index=False))

print()
print('=' * 96)
print('DEMOGRAPHIC CONCORDANCE, ALL FIVE REPORTED FIELDS')
print('=' * 96)
print(pd.DataFrame([
    summarise(set_a, 'A cross arm, one to one origin', ALL_FIELDS),
    summarise(set_b, 'B same arm, shared origin', ALL_FIELDS),
    summarise(set_c, 'C cross arm, different origin', ALL_FIELDS),
]).to_string(index=False))

print()
print('=' * 96)
print('PERMUTATION TEST WITHIN THE CANDIDATE PAIRS')
print('=' * 96)
print('Pairings are reshuffled among the same records, so the marginal')
print('distribution of every trait is held fixed.')
observed = np.mean([matches(l, r, STABLE) for l, r in set_a])
lefts = [l for l, _ in set_a]
rights = [r for _, r in set_a]
draws = np.empty(20000)
for i in range(20000):
    order = rng.permutation(len(rights))
    draws[i] = np.mean([matches(lefts[k], rights[order[k]], STABLE) for k in range(len(lefts))])
p_value = float((draws >= observed).mean())
print('observed mean matches: %.3f | permuted mean: %.3f | p = %.4f'
      % (observed, draws.mean(), p_value))

print()
print('=' * 96)
print('SUBMISSION GAP WITHIN PAIRS, HOURS')
print('=' * 96)
for label, pairs in (('A cross arm, one to one origin', set_a), ('B same arm, shared origin', set_b)):
    gaps = np.array([abs((l['submitted_at'] - r['submitted_at']).total_seconds()) / 3600
                     for l, r in pairs])
    print('%-34s n=%3d  median=%7.2f  under 1h=%4.0f%%  under 24h=%4.0f%%'
          % (label, len(gaps), np.median(gaps), 100 * (gaps < 1).mean(), 100 * (gaps < 24).mean()))

print()
print('=' * 96)
print('CANDIDATE PAIRS IN DETAIL')
print('=' * 96)
rows = []
for left, right in set_a:
    rows.append({
        'origin': left['origin'],
        'wave': left['wave'],
        'stable_matches': matches(left, right, STABLE),
        'all_matches': matches(left, right, ALL_FIELDS),
        'gap_hours': round(abs((left['submitted_at'] - right['submitted_at']).total_seconds()) / 3600, 2),
    })
detail = pd.DataFrame(rows).sort_values(['stable_matches', 'gap_hours'], ascending=[False, True])
print(detail.to_string(index=False))

OUT = outputs('clustered')
detail.to_csv(os.path.join(OUT, 'pairing_evidence.csv'), index=False)

record_run('check_pairing', {
    'arms': list(ARMS),
    'cluster': 'origin',
    'concordance_fields': STABLE,
    'all_compared_fields': ALL_FIELDS,
    'sampling_seed': SEED,
    'records': int(len(frame)),
    'clusters': int(frame['origin'].nunique()),
    'schema_digest': schema_digest(),
})

print()
print('detail written to:', os.path.join(OUT, 'pairing_evidence.csv'))
