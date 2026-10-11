# Experiment Record

## Prediction
Historical experiments compared tree-based models under rolling time validation. The strongest configurations achieved roughly R² ≈ 0.98 and WAPE ≈ 6–8% in the public dataset, depending on split and subgroup.

## Anomaly detection
A rolling-median + robust-MAD method was developed after rejecting naive actual/predicted ratios that were unstable near zero baselines.

## Robustness
Priority rankings were tested under alternative peer definitions, climate/site normalization, bootstrap resampling and multi-indicator screening.

## Counterfactual
Expected-use screening is treated as a prioritization signal rather than a causal effect. Placebo/falsification checks are reserved for field-validation data. High apparent effects near low baselines should be treated as unstable rather than automatically accepted.

## Intervention
Intervention families were mapped from temporal/behavioral fingerprints. Their effect rates are scenario assumptions, not measured engineering savings.

## Carbon
Carbon scenarios use an emission factor only conditionally. The public derivative's meter-unit lineage must be verified before treating readings as kWh.

## Field validation
The validation prototype now supports treatment/control DID, a simple pre-trend slope diagnostic, a pre-period placebo DID and clustered bootstrap intervals. These are evidence safeguards, not proof of causality; weather, seasonality, autocorrelation and intervention compliance still require study-specific treatment.

## Generalization to unseen buildings

The reproducible pipeline now evaluates a second split in addition to chronological holdout. `scripts/run_v3.py` runs a `GroupShuffleSplit` by `building_id`, fits a separate ExtraTrees model on training buildings only, and writes the results under `unseen_building_evaluation` in `outputs/metrics.json`. It also writes row-level predictions to `outputs/unseen_building_predictions.csv`.

Run it with the same input data as the main pipeline:

```bash
python scripts/run_v3.py --input data/test.csv --output outputs --building-test-size 0.2 --building-seed 42
```

This is a transfer-to-new-buildings test **with historical meter data available for each test building**: past-only lag/rolling features are retained, but the held-out buildings are excluded from model fitting. It is not a zero-history cold-start evaluation. Compare the unseen-building R², MAE and WAPE against the chronological metrics before presenting the model as generalizable. Synthetic unit tests verify that building IDs do not overlap across partitions; they do not substitute for running the evaluation on the full dataset. No unseen-building performance number is claimed until that run has been completed and inspected.
