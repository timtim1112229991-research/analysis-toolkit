# -*- coding: utf-8 -*-
"""Generate blinded coding sheets and the codebook for two independent coders.

Sheets carry no arm or wave label. The design is rejoined only after coding is
closed and agreement has been computed. Everything written here stays local.
"""

import os
import sys

import pandas as pd

from _paths import SCHEMA, outputs, record_run, schema_digest, workbooks

sys.stdout.reconfigure(encoding='utf-8')

from src.coding import blind_keys, build_sheet  # noqa: E402
from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402

import _coding_frames as frames  # noqa: E402

OUT = outputs('coding')

schema = Schema.from_json(SCHEMA)
paths = workbooks()
records = load_study(paths, schema)

NULLISH = {'nan', ''}
summary = []

for name, frame in frames.FRAMES.items():
    subset = records
    if name.endswith('multi_agent'):
        subset = records[records['arm'] == 'multi_agent']
    elif name.endswith('single_model'):
        subset = records[records['arm'] == 'single_model']

    sheet = build_sheet(subset, frame, shuffle_seed=20260902)
    sheet = sheet[~sheet['text'].str.lower().isin(NULLISH)].reset_index(drop=True)

    prefix = ''.join(word[0] for word in name.split('_')).upper()
    blinded, crosswalk = blind_keys(sheet, prefix=prefix)

    for coder in ('coder_a', 'coder_b'):
        path = os.path.join(OUT, f'{name}__{coder}.csv')
        blinded.to_csv(path, index=False, encoding='utf-8-sig')

    crosswalk.to_csv(os.path.join(OUT, f'{name}__crosswalk.csv'),
                     index=False, encoding='utf-8-sig')

    book = frame.as_table()
    book.insert(0, 'field', frame.field)
    book.insert(1, 'question', frame.question)
    book.to_csv(os.path.join(OUT, f'{name}__codebook.csv'), index=False, encoding='utf-8-sig')

    summary.append({'frame': name, 'codes': len(frame.keys()), 'records_to_code': len(sheet)})

overview = pd.DataFrame(summary)
overview.to_csv(os.path.join(OUT, 'coding_overview.csv'), index=False, encoding='utf-8-sig')

record_run('build_coding_sheets', {
    'frames': sorted(frames.FRAMES),
    'coders': 2,
    'blinded': True,
    'records_available': int(len(records)),
    'schema_digest': schema_digest(),
})

print(overview.to_string(index=False))
print()
print('sheets and codebooks written to:', OUT)
print('two copies of each sheet were produced, one per coder, with no arm label present')
