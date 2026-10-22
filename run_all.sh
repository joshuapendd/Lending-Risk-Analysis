#!/usr/bin/env bash
# Rebuild everything from the raw CSV. Assumes .venv exists (see README).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/load_data.py
$PY src/run_cohorts.py
$PY src/model.py
$PY -W ignore src/fair_lending.py | tee outputs/fair_lending_report.txt
$PY -W ignore src/verify.py
