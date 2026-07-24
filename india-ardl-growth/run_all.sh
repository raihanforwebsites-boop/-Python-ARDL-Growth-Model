#!/usr/bin/env bash
# Runs the full pipeline end-to-end, in order.
set -e
cd "$(dirname "$0")"

python src/00_generate_data.py
python src/01_descriptive_stats.py
python src/02_stationarity_tests.py
python src/03_ardl_model.py
python src/04_diagnostics.py
