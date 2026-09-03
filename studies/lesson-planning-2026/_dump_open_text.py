# -*- coding: utf-8 -*-
"""Dump the open response fields for inductive coding frame development.

Output stays local. It is written to the parent data folder, not to the
repository, and is never released.
"""

import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
from _paths import workbooks

ARM = {'360254564': 'multi_agent', '367144800': 'multi_agent',
       '361176326': 'single_model', '367144889': 'single_model'}
FIELDS = {24: 'open_positive', 25: 'open_negative', 26: 'open_process'}

rows = []
for path in workbooks():
    name = os.path.basename(path)
    arm = next((v for k, v in ARM.items() if k in name), 'unassigned')
    raw = pd.read_excel(path)
    for index, column in FIELDS.items():
        for i, value in enumerate(raw.iloc[:, index]):
            rows.append({'arm': arm, 'source': name.split('_')[0],
                         'record': i + 1, 'field': column,
                         'text': str(value).strip()})

frame = pd.DataFrame(rows)
frame['length'] = frame['text'].str.len()

for field in FIELDS.values():
    print('=' * 100)
    print('FIELD:', field)
    print('=' * 100)
    subset = frame[frame['field'] == field]
    for arm in ('multi_agent', 'single_model'):
        print('--- %s ---' % arm)
        for _, row in subset[subset['arm'] == arm].iterrows():
            print('[%s:%02d] %s' % (row['source'], row['record'], row['text']))
        print()
