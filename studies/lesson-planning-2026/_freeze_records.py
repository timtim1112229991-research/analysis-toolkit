# -*- coding: utf-8 -*-
"""Freeze the analysable record set and record checksums.

The manuscript will assert that every reported number came from one run over one
fixed set of records. That assertion is only checkable if the records carry a
digest taken before the run. This script writes that digest, and re-run with
--verify it reports whether the current files still match it.

Two digests are recorded. The file digest covers the source workbooks byte for
byte and detects any edit to the collection instances. The frame digest covers
the harmonised analysable table and detects a change in loading, mapping or
derivation even where the source files are untouched.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

from _paths import SCHEMA, SNAPSHOT, workbooks

from src.loading import load_study  # noqa: E402
from src.schema import Schema  # noqa: E402


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(131072), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame, items):
    """Digest of the analysable table under a stable row and column order."""
    columns = ['arm', 'wave', 'origin', 'submitted_at', 'duration_s'] + items
    present = [c for c in columns if c in frame.columns]
    ordered = frame[present].sort_values(
        [c for c in ('submitted_at', 'origin', 'arm') if c in present],
        kind='mergesort').reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator='\n').encode('utf-8')
    return hashlib.sha256(payload).hexdigest(), present


def describe():
    schema = Schema.from_json(SCHEMA)
    paths = workbooks()
    frame = load_study(paths, schema)
    items = schema.item_names()
    digest, columns = frame_digest(frame, items)

    return {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'purpose': ('Fixes the record set behind the reported estimates so that a later run '
                    'can be shown to have used exactly these records.'),
        'source_files': [
            {
                'name': os.path.basename(path),
                'bytes': os.path.getsize(path),
                'sha256': file_digest(path),
            }
            for path in paths
        ],
        'schema': {
            'name': os.path.basename(SCHEMA),
            'sha256': file_digest(SCHEMA),
            'items': len(items),
        },
        'analysable_frame': {
            'records': int(len(frame)),
            'columns_hashed': columns,
            'sha256': digest,
            'by_arm': {str(k): int(v) for k, v in frame['arm'].value_counts().sort_index().items()},
            'by_wave': {str(k): int(v) for k, v in frame['wave'].value_counts().sort_index().items()},
            'origins': int(frame['origin'].nunique()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Freeze or verify the record set')
    parser.add_argument('--verify', action='store_true',
                        help='compare the current files against the stored snapshot')
    arguments = parser.parse_args()

    current = describe()

    if not arguments.verify:
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        with open(SNAPSHOT, 'w', encoding='utf-8') as handle:
            json.dump(current, handle, indent=2)
        print('snapshot written to:', SNAPSHOT)
        print('records: %d across %d origins'
              % (current['analysable_frame']['records'], current['analysable_frame']['origins']))
        for entry in current['source_files']:
            print('  %s  %s' % (entry['sha256'][:16], entry['name']))
        print('frame digest: %s' % current['analysable_frame']['sha256'])
        return 0

    if not os.path.exists(SNAPSHOT):
        print('no snapshot exists; run without --verify first')
        return 2

    with open(SNAPSHOT, encoding='utf-8') as handle:
        stored = json.load(handle)

    problems = []
    stored_files = {e['name']: e['sha256'] for e in stored['source_files']}
    current_files = {e['name']: e['sha256'] for e in current['source_files']}
    for name in sorted(set(stored_files) | set(current_files)):
        if stored_files.get(name) != current_files.get(name):
            problems.append('source file changed or missing: %s' % name)
    if stored['schema']['sha256'] != current['schema']['sha256']:
        problems.append('measurement schema changed')
    if stored['analysable_frame']['sha256'] != current['analysable_frame']['sha256']:
        problems.append('analysable frame changed')
    if stored['analysable_frame']['records'] != current['analysable_frame']['records']:
        problems.append('record count changed')

    if problems:
        print('VERIFICATION FAILED')
        for problem in problems:
            print('  -', problem)
        return 1

    print('VERIFIED: %d records match the snapshot taken %s'
          % (current['analysable_frame']['records'], stored['created_at']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
