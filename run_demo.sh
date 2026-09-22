#!/usr/bin/env bash
set -e
python scripts/run_v3.py --input data/test.csv --output outputs
streamlit run app/streamlit_app.py
