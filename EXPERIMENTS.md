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
The validation UI supports treatment/control DID, a pre-intervention slope diagnostic, a placebo DID restricted to observations before the actual intervention, and an optional cluster bootstrap resampled separately within treated and control groups. These are evidence safeguards, not proof of causality; weather, seasonality, autocorrelation and intervention compliance still require study-specific treatment.

## Generalization to unseen buildings

The reproducible pipeline now evaluates a second split in addition to chronological holdout. `scripts/run_v3.py` runs a `GroupShuffleSplit` by `building_id`, fits a separate ExtraTrees model on training buildings only, and writes the results under `unseen_building_evaluation` in `outputs/metrics.json`. It also writes row-level predictions to `outputs/unseen_building_predictions.csv`.

Run it with the same input data as the main pipeline:

```bash
python scripts/run_v3.py --input data/test.csv --output outputs --building-test-size 0.2 --building-seed 42
```

This is a transfer-to-new-buildings test **with historical meter data available for each test building**: past-only lag/rolling features are retained, but the held-out buildings are excluded from model fitting. It is not a zero-history cold-start evaluation. Compare the unseen-building R², MAE and WAPE against the chronological metrics before presenting the model as generalizable. Synthetic unit tests verify that building IDs do not overlap across partitions; they do not substitute for running the evaluation on the full dataset. No unseen-building performance number is claimed until that run has been completed and inspected.


## Building-level error diagnostics

Chronological test predictions are summarized per building with test-row count, MAE, RMSE, WAPE, mean residual (actual minus predicted), bias direction and median APE. APE excludes zero actual readings; WAPE and APE are marked undefined when their denominators are invalid. Reports list the highest-MAE building for the current split and link the full diagnostic CSV plus top-10 export. These are error diagnostics, not a ranking of operational waste or retrofit value.

## Fair baseline comparison

The pipeline exports two complementary views: (1) each baseline's own valid predictions on chronological test dates, with sample count and coverage; and (2) a fair comparison that intersects rows with finite ExtraTrees and all three baseline predictions, then scores every method on that identical cohort. Only the second view supports a direct model-versus-baseline comparison. Synthetic tests verify cohort alignment; full-data results still require a fresh run on the actual dataset.


## Carbon factor and annualization audit

The UI uses the official 2023 Chinese national average electricity CO2 emission factor (0.5306 kgCO2/kWh) and the 2023 Guangdong average factor (0.4419 kgCO2/kWh) for location-based screening. These are electricity CO2 emission factors, so the scenario output is labelled CO2 rather than CO2e. The national factor excluding market-traded non-fossil electricity is a different value (0.6096 kgCO2/kWh) and should only be used when that accounting boundary is appropriate. Official source: https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/t20251231_1139517.html

When uploaded data cover less than a full year, the dashboard now annualizes the observed energy total by dividing by the number of observed daily records and multiplying by 365. This is a simple average-daily extrapolation, not a seasonal or weather-adjusted forecast; sparse or unrepresentative periods can bias the result. Meter units must still be verified before interpreting the output as kWh.
