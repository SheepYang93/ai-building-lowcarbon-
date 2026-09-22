# AI-driven Building Energy Anomaly & Low-carbon Decision System

A portfolio project applying machine learning to building-energy anomaly detection, counterfactual analysis, intervention screening, and low-carbon decision support.

## Pipeline
Raw meter data → data quality control → lag/rolling features → strict chronological ML prediction → robust anomaly detection → building energy fingerprint → counterfactual / intervention analysis → parameterized carbon scenario → decision dashboard.

## Reproducibility
The project is designed around a BDG2-derived daily building-energy dataset. The exact unit lineage of `meter_reading` must be verified before calling the values kWh.

```bash
pip install -r requirements.txt
python scripts/run_v3.py --input data/test.csv --output outputs
streamlit run app/streamlit_app.py
```

Place the real dataset at `data/test.csv`. Raw datasets are intentionally excluded from Git.

## Model
ExtraTreesRegressor with chronological train/test validation. Historical energy features are shifted before rolling calculations to avoid future leakage. The demo uses a deterministic training cap for ordinary hardware.

## Important limitations
- Scenario CO2e is not measured emissions.
- Carbon calculations require a verified energy unit and emission factor.
- Public-data screening is not evidence of savings in a real campus.
- Field validation requires treatment/control buildings and before/after data.

## Portfolio positioning
This is an AI/data-engineering portfolio project rather than a grant application.

**prediction → anomaly detection → counterfactual reasoning → intervention screening → validation**

The system is deliberately explicit about uncertainty and evidence boundaries.