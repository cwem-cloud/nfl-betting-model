"""Run the full Pocket Capper slate (what the dashboard's Run button does).

    python scripts/run_slate.py                 # NFL + CFB, with props
    python scripts/run_slate.py --sports nfl --no-props
"""
import argparse

import _path  # noqa: F401

from pocket_capper import db
from pocket_capper.engine import grading, slate

ap = argparse.ArgumentParser()
ap.add_argument("--sports", default="nfl,cfb")
ap.add_argument("--no-props", action="store_true")
args = ap.parse_args()

grading.grade_all()
res = slate.run(tuple(args.sports.split(",")), include_props=not args.no_props, progress=print)
print(f"run {res['run_id']}: {res['picks']} plays written; errors: {res['errors'] or 'none'}")
db.compact()
