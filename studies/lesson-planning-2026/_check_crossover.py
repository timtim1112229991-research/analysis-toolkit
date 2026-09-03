# -*- coding: utf-8 -*-
"""Test whether the two arms drew on independent respondents.

Independence is assumed by the unpaired comparison. Shared network origins and
verbatim identical open responses across arms would falsify that assumption.
"""

import os
import sys
from difflib import SequenceMatcher

import pandas as pd

from _paths import record_run, workbooks

sys.stdout.reconfigure(encoding='utf-8')

ARM = {'360254564': 'multi_agent', '367144800': 'multi_agent',
       '361176326': 'single_model', '367144889': 'single_model'}
WAVE = {'360254564': 'wave_1', '361176326': 'wave_1',
        '367144800': 'wave_2', '367144889': 'wave_2'}
FIELDS = {24: 'open_positive', 25: 'open_negative', 26: 'open_process'}
# Placeholder answers that carry no content. Escaped so that this file
# stays ASCII, and glossed here so the set can be read:
#   \u65e0         none
#   \u7121         none, traditional form
#   \u6ca1\u6709   there is not
#   \u6682\u65e0   none for now
#   \u5426         no
#   \u3002         a full stop alone
#   1              a bare digit
#   nan            a missing cell
#   (empty)        an empty string
#   \u4e0d\u77e5\u9053 do not know
#   \u6ca1         not
NULLISH = {'\u65e0', '\u7121', '\u6ca1\u6709', '\u6682\u65e0', '\u5426', '\u3002',
           '1', 'nan', '', '\u4e0d\u77e5\u9053', '\u6ca1'}

rows = []
for path in workbooks():
    name = os.path.basename(path)
    code = name.split('_')[0]
    raw = pd.read_excel(path)
    for i in range(len(raw)):
        record = {'code': code, 'arm': ARM[code], 'wave': WAVE[code], 'record': i + 1,
                  'ip': str(raw.iloc[i, 5]).split('(')[0].strip(),
                  'submitted': raw.iloc[i, 1]}
        for index, field in FIELDS.items():
            record[field] = str(raw.iloc[i, index]).strip()
        rows.append(record)

frame = pd.DataFrame(rows)

print('=' * 96)
print('1. NETWORK ORIGINS SHARED ACROSS ARMS')
print('=' * 96)
by_ip = frame.groupby('ip')['arm'].nunique()
both = by_ip[by_ip > 1].index.tolist()
print('distinct origins: %d | origins appearing in BOTH arms: %d'
      % (frame['ip'].nunique(), len(both)))
affected = frame[frame['ip'].isin(both)]
print('records at a cross-arm origin: %d of %d (%.0f%%)'
      % (len(affected), len(frame), 100 * len(affected) / len(frame)))
print()
print(pd.crosstab(affected['wave'], affected['arm']))
print()
print(frame[frame['ip'].isin(both)].groupby(['wave', 'ip', 'arm']).size().to_frame('records').to_string())

print()
print('=' * 96)
print('2. VERBATIM IDENTICAL OPEN RESPONSES ACROSS ARMS')
print('=' * 96)
for field in FIELDS.values():
    substantive = frame[~frame[field].isin(NULLISH) & (frame[field].str.len() > 6)]
    grouped = substantive.groupby(field)['arm'].nunique()
    shared = grouped[grouped > 1].index.tolist()
    print('%s: %d substantive answers, %d appear verbatim in both arms'
          % (field, len(substantive), len(shared)))
    for text in shared:
        who = substantive[substantive[field] == text][['code', 'record', 'arm', 'wave']]
        print('   length %d characters, submitted by:' % len(text))
        print(who.to_string(index=False))

print()
print('=' * 96)
print('3. NEAR-DUPLICATE ANSWERS ACROSS ARMS (similarity above 0.80)')
print('=' * 96)
for field in FIELDS.values():
    subset = frame[~frame[field].isin(NULLISH) & (frame[field].str.len() > 8)]
    a = subset[subset['arm'] == 'multi_agent']
    b = subset[subset['arm'] == 'single_model']
    hits = 0
    for _, left in a.iterrows():
        for _, right in b.iterrows():
            ratio = SequenceMatcher(None, left[field], right[field]).ratio()
            if ratio > 0.80:
                hits += 1
                print('%s | %.2f | %s:%02d  vs  %s:%02d'
                      % (field, ratio, left['code'], left['record'],
                         right['code'], right['record']))
    if hits == 0:
        print('%s: none' % field)

print()
print('=' * 96)
print('4. WAVE 2 STRUCTURE')
print('=' * 96)
w2 = frame[frame['wave'] == 'wave_2']
print('records per arm:')
print(w2.groupby('arm').size().to_string())
print()
print('origins by arm:')
print(w2.groupby(['ip', 'arm']).size().to_frame('records').to_string())

print()
print('=' * 96)
print('5. NULL ANSWER RATE BY FIELD AND ARM')
print('=' * 96)
for field in FIELDS.values():
    null_mask = frame[field].isin(NULLISH) | (frame[field].str.len() <= 1)
    table = frame.assign(is_null=null_mask).groupby('arm')['is_null'].agg(['sum', 'count'])
    table['percentage'] = (100 * table['sum'] / table['count']).round(1)
    print('\n%s' % field)
    print(table.to_string())

record_run('check_crossover', {
    'open_text_fields': sorted(FIELDS.values()),
    'shared_origin_test': 'network origin recurring across arms',
    'verbatim_test': 'sequence similarity between open responses across arms',
    'records': int(len(frame)),
})
