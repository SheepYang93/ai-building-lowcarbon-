# Technical Review & Interview Notes

## Project status (2026-11-12)

**Project:** AI Building Energy Anomaly Diagnosis & Low-carbon Decision Support System  
**Positioning:** AI + building-energy analytics + low-carbon decision-support prototype.

## Three-stage delivery status

| Stage | Status | Evidence / remaining work |
|---|---|---|
| 1. Runnable & accessible | Local runnable path documented; public deployment pending | Owner must deploy via Streamlit Community Cloud and complete smoke tests |
| 2. Verifiable & explainable | Evaluation design and evidence boundaries documented | Run pipeline against verified data; inspect generated metrics and baseline comparison; confirm units |
| 3. Interview-ready & transferable | Resume card and portfolio walkthrough exist | Review this note, prepare explanations, capture real app screenshots after deployment |

## Deployment checklist (owner action required)

1. Open Streamlit Community Cloud and sign in with GitHub.
2. Create app from repository `SheepYang93/ai-building-lowcarbon-`.
3. Branch: `main`; entrypoint: `app/streamlit_app.py`.
4. Wait for build; fix dependency or startup errors from logs.
5. Upload `examples/synthetic_building_energy.csv` and test the complete flow.
6. Verify prediction, anomaly diagnosis, report download, conditional Scope 2 labeling and DID input.
7. Capture screenshots from the running app. Label synthetic-data screenshots as **synthetic demo**, not real building evidence.

## Evidence checklist before quantitative claims

- [ ] Identify dataset source, license, preprocessing and meter units.
- [ ] Confirm electricity readings are kWh before carbon conversion.
- [ ] Run chronological evaluation and save generated metrics.
- [ ] Compare ExtraTrees and naive-1-day / seasonal-7-day / rolling-median baselines on identical held-out rows.
- [ ] Record MAE, WAPE, R², split and sample count.
- [ ] Run building-group holdout; test buildings have historical meter data (not cold-start).
- [ ] Inspect errors by building/time and check whether results are dominated by a few buildings.
- [ ] Treat anomaly detection as candidate screening; inspect false positives and operating context.
- [ ] Treat counterfactual/intervention values as hypotheses, not measured savings.
- [ ] Treat Scope 2 outputs as conditional until unit, factor year/geography and boundary are verified.
- [ ] Treat DID as preliminary; inspect trends, placebo, uncertainty, weather/seasonality and intervention compliance.

## Interview explanation (90 seconds)

I built a Python and Streamlit prototype for building-energy diagnosis and low-carbon decision support. It starts from daily meter data, applies data checks and past-only lag/rolling features, and uses ExtraTrees to estimate expected energy use under chronological validation. A rolling median and robust MAD identify anomaly candidates; temporal behavior is summarized into an energy fingerprint that can support intervention screening. The system then provides conditional Scope 2 scenarios and a treatment/control DID workflow for later field evaluation. I deliberately distinguish prediction, screening and field evidence: the prototype does not claim verified energy savings. The next validation step is to verify meter units and dataset lineage, compare against simple baselines on identical held-out rows, and test on real building data.

## Technical questions to prepare

### Why chronological validation?
Random splitting can allow later observations to inform evaluation of earlier periods. A chronological holdout better reflects forecasting into a future period. Lag and rolling features must be shifted so the target timestamp does not leak into its own features.

### Why ExtraTrees?
It handles nonlinear relationships and feature interactions with relatively little feature scaling. It is a pragmatic baseline, not proof that it is the best model; comparison with simple baselines and other candidates is necessary.

### Why robust MAD for anomalies?
Rolling median/MAD is less sensitive to extreme observations than mean/standard deviation. An anomaly is only a candidate requiring context—not automatically equipment failure or avoidable waste.

### Does a high R² prove energy savings?
No. Forecast accuracy measures prediction; savings require a defined intervention and credible counterfactual/field design.

### What does DID establish?
DID estimates a treatment-control change contrast under assumptions such as parallel trends. Pre-trend, placebo and bootstrap checks help, but cannot guarantee causality; weather, seasonality, autocorrelation and compliance need attention.

### Biggest limitation?
The current work is a decision-support prototype. Verified meter-unit lineage, real-building deployment, field intervention data and measured savings are not yet established.

## Resume and portfolio guardrails

Use: “implemented”, “built”, “evaluated”, “screened”, “estimated conditionally”, “designed validation workflow”. Use exact metrics only when generated from a named dataset/split and reproducible artifact. Mark synthetic screenshots/data as “synthetic demo”.

Avoid: “achieved X% energy savings” without field measurement; “proved causal impact” based only on a demo DID chart; “deployed/online” before a public app URL is live and smoke-tested; treating generated concept art as a real application screenshot.
