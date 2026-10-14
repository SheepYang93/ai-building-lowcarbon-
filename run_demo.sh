#!/usr/bin/env bash
set -e
if [ ! -f "data/test.csv" ]; then
  echo "Missing data/test.csv. See data/README.md for dataset instructions, or run the Streamlit UI directly with: streamlit run app/streamlit_app.py"
  exit 2
fi
python scripts/run_v3.py --input data/test.csv --output outputs
streamlit run app/streamlit_app.py
