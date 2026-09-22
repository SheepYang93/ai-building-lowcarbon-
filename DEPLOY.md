# Deploy the Streamlit demo

The repository is ready for deployment to Streamlit Community Cloud.

## Steps

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app.
3. Select repository `SheepYang93/ai-building-lowcarbon-`.
4. Branch: `main`.
5. Main file: `app/streamlit_app.py`.
6. Deploy.

The app is designed to remain useful without the original raw dataset: curated public-data example artifacts are included under `examples/`. If `outputs/` exists after running the local pipeline, the dashboard will prefer those generated results.

### Important

This repository does not contain the raw `test.csv`. The online demo therefore uses the included example artifacts unless generated pipeline outputs are supplied.

CO₂e values shown by the demo are conditional scenarios. Verify meter units and emission factors before interpreting them as real emissions.
