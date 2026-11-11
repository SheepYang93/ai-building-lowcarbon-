# Deploy the Demo on Streamlit Community Cloud

Railway deployment is currently unavailable because the project workspace's Railway trial has expired. This repository can be deployed on Streamlit Community Cloud as a free public demo (subject to Streamlit's current usage and availability policies).

## Deploy

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Select **Create app** / **Deploy an app**.
3. Choose repository: `SheepYang93/ai-building-lowcarbon-`.
4. Branch: `main`.
5. Main file path: `app/streamlit_app.py`.
6. Advanced settings: leave secrets empty unless the app later requires them; select a supported Python version if prompted.
7. Deploy and wait for the build logs to finish.

## Smoke test after deployment

Use the included synthetic dataset; it is explicitly for UI and workflow smoke testing, not real energy-saving evidence.

- [ ] App opens without an import or dependency error.
- [ ] Overview and navigation render correctly.
- [ ] Upload `examples/synthetic_building_energy.csv` and complete the diagnosis workflow.
- [ ] Prediction/anomaly/fingerprint outputs render.
- [ ] Carbon scenario is clearly marked conditional.
- [ ] Downloadable report works.
- [ ] Field-validation page accepts treatment/control sample data and labels results as preliminary.
- [ ] Mobile layout is usable; no clipped sidebar or tables.

## If deployment fails

- **Dependency/build error:** inspect build logs; keep `requirements.txt` minimal and pin only dependencies that need reproducibility.
- **Missing file:** confirm the main file path is exactly `app/streamlit_app.py` and the repository branch is `main`.
- **Memory / slow startup:** avoid loading large training artifacts during app startup; use the small included demo artifacts.
- **Secrets:** never put tokens or credentials in source files; use Streamlit Cloud's Secrets interface only if needed.

## Portfolio status wording

Until the app is deployed and smoke-tested, describe it as **“Streamlit demo (deployment pending)”** and link to the repository plus this reproducible local command:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Do not label a URL as a live demo if it is unavailable.
