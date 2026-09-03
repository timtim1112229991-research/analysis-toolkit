# -*- coding: utf-8 -*-
"""Reconstruct the session timeline within confirmed pairs.

Each record carries a submission time and the seconds the respondent spent on
the form. The two together give a start time, so the two questionnaires filled
at one origin can be placed on a line and checked for overlap. Overlapping fill
windows would mean both forms were open at once, which is incompatible with
using one tool, reporting on it, then using the other.
"""

import os
import sys

import pandas as pd

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

STABLE = ['experience', 'stage', 'subject']
ARMS = ('multi_agent', 'single_model')

schema = Schema.from_json(SCHEMA)
paths = workbooks()
frame = load_study(paths, schema).reset_index(drop=True)

counts = frame.groupby(['origin', 'arm']).size().unstack(fill_value=0)
one_to_one = counts[(counts.get(ARMS[0], 0) == 1) & (counts.get(ARMS[1], 0) == 1)].index

rows = []
for origin in one_to_one:
    block = frame[frame['origin'] == origin]
    a = block[block['arm'] == ARMS[0]].iloc[0]
    b = block[block['arm'] == ARMS[1]].iloc[0]
    if sum(1 for f in STABLE if str(a[f]) == str(b[f])) != 3:
        continue

    first, second = (a, b) if a['submitted_at'] <= b['submitted_at'] else (b, a)
    gap = (second['submitted_at'] - first['submitted_at']).total_seconds()
    second_started = gap - float(second['duration_s'])

    rows.append({
        'origin': origin,
        'first_arm': first['arm'],
        'first_duration_s': int(first['duration_s']),
        'gap_s': int(gap),
        'second_duration_s': int(second['duration_s']),
        'second_started_after_first_submit_s': int(second_started),
        'windows_overlap': second_started < 0,
        'idle_between_s': int(second_started) if second_started >= 0 else 0,
    })

table = pd.DataFrame(rows)
print('=' * 100)
print('SESSION TIMELINE WITHIN CONFIRMED PAIRS')
print('=' * 100)
print(table.to_string(index=False))

print()
print('=' * 100)
print('SUMMARY')
print('=' * 100)
print('confirmed pairs examined:                       %3d' % len(table))
print('pairs whose two forms were open at once:        %3d' % int(table['windows_overlap'].sum()))
print('median gap between submissions, seconds:        %3d' % int(table['gap_s'].median()))
print('median time on the second form, seconds:        %3d' % int(table['second_duration_s'].median()))
print('median idle time between the two forms, seconds:%3d'
      % int(table.loc[~table['windows_overlap'], 'idle_between_s'].median()
            if (~table['windows_overlap']).any() else -1))
print()
print('A respondent who used a second tool between the two forms would leave a')
print('long idle interval. A short or negative interval means no second tool was')
print('used in between.')

print()
print('=' * 100)
print('TIME ON FORM, ALL RECORDS, BY ARM')
print('=' * 100)
print(frame.groupby('arm')['duration_s'].describe()[['count', 'min', '25%', '50%', '75%', 'max']].to_string())

OUT = outputs('clustered')
table.to_csv(os.path.join(OUT, 'session_timeline.csv'), index=False)

record_run('check_sessions', {
    'arms': list(ARMS),
    'cluster': 'origin',
    'concordance_fields': STABLE,
    'pairs_reconstructed': int(len(table)),
    'records': int(len(frame)),
    'schema_digest': schema_digest(),
})

print()
print('written to:', os.path.join(OUT, 'session_timeline.csv'))

