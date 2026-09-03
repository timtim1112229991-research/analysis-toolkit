# -*- coding: utf-8 -*-
"""Locations this study reads from and writes to.

The response records are held in restricted storage and are not part of this
repository, so the scripts here cannot run end to end from a clone alone. What
a reader can do is inspect exactly how each reported quantity was produced, and
run the same code against their own records by pointing the two environment
variables below at them.

    STUDY_DATA      directory holding the source workbooks
    STUDY_OUTPUTS   directory to write results into

Left unset, both fall back to the layout used for the reported run, which
places this repository beside the data folder and the manuscript folder.
"""

import os
import sys

STUDY = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(os.path.dirname(STUDY))
WORKSPACE = os.path.dirname(REPOSITORY)

# The drivers import the toolkit as ``src.*``, so the repository root has to be
# importable before any of them runs.
if REPOSITORY not in sys.path:
    sys.path.insert(0, REPOSITORY)

DATA = os.environ.get('STUDY_DATA') or os.path.join(WORKSPACE, 'source-data')
OUTPUTS = os.environ.get('STUDY_OUTPUTS') or os.path.join(WORKSPACE, 'Paper1', 'outputs')

SCHEMA = os.path.join(STUDY, 'study.schema.json')
SNAPSHOT = os.path.join(STUDY, 'record_snapshot.json')
MANIFESTS = os.path.join(STUDY, 'manifests')


def workbooks():
    """The source workbooks, in the fixed order the reported run used."""
    if not os.path.isdir(DATA):
        raise SystemExit(
            'No record directory at %s. Set STUDY_DATA to the folder holding '
            'the source workbooks.' % DATA)
    return [os.path.join(DATA, name) for name in sorted(os.listdir(DATA))
            if name.endswith('.xlsx')]


def outputs(*parts):
    """A directory under the results tree, created if it is not there yet."""
    path = os.path.join(OUTPUTS, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def schema_digest():
    """Digest of the configuration in force, so a run can be tied to one layout."""
    import hashlib

    with open(SCHEMA, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def record_run(driver, parameters):
    """Write the provenance record for one driver.

    Results are held back from this repository but their provenance is not, so
    these records are committed. Each states the code state, interpreter and
    parameter values behind one script, which is what lets a reported number be
    traced to the run that produced it.
    """
    from pathlib import Path

    from src.reporting import run_manifest

    return run_manifest(Path(MANIFESTS), parameters, name=driver.lstrip('_'))
