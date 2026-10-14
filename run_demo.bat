@echo off
if not exist "data\test.csv" (
  echo Missing data\test.csv. See data\README.md for dataset instructions.
  echo To open the UI without retraining, run: streamlit run app\streamlit_app.py
  exit /b 2
)
python scripts\run_v3.py --input data\test.csv --output outputs
if errorlevel 1 exit /b %errorlevel%
streamlit run app\streamlit_app.py
