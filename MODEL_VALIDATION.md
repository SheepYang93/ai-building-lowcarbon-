# Model Validation Protocol

This document describes the reproducible validation currently implemented in `src/pipeline.py` and `scripts/run_v3.py`. It is a protocol and evidence boundary—not a claim that a fresh full-data run has already been completed.

## 1. Chronological holdout

- Rows are sorted by building and date; features use calendar fields, exact calendar-day lags, and prior-calendar-day rolling windows; missing dates remain missing rather than compressing time.
- A date cutoff is selected at 80% of the distinct dates with complete features.
- Training rows are strictly earlier than the cutoff; test rows are on or after it.
- ExtraTrees is fit only on the training partition and evaluated on the later test partition.
- The deterministic train cap is applied only to training rows.

## 2. Baseline models

Three past-only baselines are evaluated:

| Baseline | Prediction rule |
|---|---|
| Naive-1-day | Same building's exact previous calendar-day reading |
| Seasonal-7-day | Same building's exact reading seven calendar days earlier |
| Rolling median-28 | Median of available readings in the preceding 28 calendar days; at least 7 observations required |

Missing history produces an unavailable prediction; it is not silently imputed. Duplicate building/date records are rejected after date normalization.

## 3. Fair comparison cohort

The fair comparison is the intersection of chronological test rows for which ExtraTrees and all three baselines have finite predictions. ExtraTrees and every baseline are scored on this identical cohort. The report records its row count and coverage relative to the chronological test set.

The separate baseline table may use each baseline's own available test rows; it must not be used as the headline model-vs-baseline comparison.

## 4. Building-group holdout

A separate `GroupShuffleSplit` partitions by `building_id`. The model is fit only on training buildings; test building IDs are disjoint from training IDs.

**Interpretation boundary:** test buildings may contribute their own historical meter observations to lag/rolling features. This evaluates transfer to held-out buildings with history, not a zero-history cold-start scenario.

## 5. Metrics and diagnostics

Anomaly screening uses the configured relative-gap and robust-z thresholds; zero-baseline periods are excluded from relative anomaly classification. The pipeline reports R², MAE, RMSE and WAPE with explicit undefined states: WAPE is undefined when the sum of absolute actual values is zero; R² is undefined for a one-row cohort. Empty fair-comparison cohorts are represented as `no_shared_valid_rows` instead of crashing. Non-finite metric values are serialized as JSON `null` and displayed as `N/A` in the Markdown report and CSV exports.

Building-level diagnostics include test-row count, MAE, RMSE, WAPE, mean residual (actual minus predicted), bias direction (positive = underprediction; negative = overprediction), median APE and the number of valid APE rows. APE excludes zero actual readings rather than dividing by an artificial epsilon. The pipeline exports chronological row-level predictions/residuals, a full building error table and a top-10 high-MAE table. Metrics depend on dataset filtering, period, building composition and split settings; they are not universal performance guarantees.

## 6. Leakage checks

Automated tests cover:
- lag features use prior observations rather than the current target;
- baseline predictions use exact calendar offsets and do not cross building boundaries;
- changing future meter readings leaves earlier engineered features unchanged;
- baseline predictions are past-only under future-value perturbation;
- building-group train/test IDs are disjoint;
- fair comparison metrics use a common valid-row count.

These tests reduce known leakage risks; they do not prove the absence of every possible data, preprocessing or deployment leakage.

## 7. Reproduce

```bash
python scripts/run_v3.py --input PATH_TO_DATA.csv --output outputs --building-test-size 0.2 --building-seed 42
```

The run also writes `outputs/run_manifest.json` with UTC timestamp, input CSV SHA-256 (to identify the exact dataset file), filtered row/building/date coverage, run parameters, Python and core package versions, optional Git commit, and artifact index. The manifest records the input filename only—not its local absolute path. Inspect `outputs/metrics.json`, `outputs/evaluation_report.md` (Markdown), `outputs/evaluation_report.docx` (Word), `outputs/chronological_test_predictions.csv`, `outputs/unseen_building_predictions.csv`, `outputs/error_analysis_by_building.csv` and `outputs/top10_high_error_buildings.csv`.

The repository's CI uses synthetic fixtures for fast regression tests. Synthetic test success is not a substitute for a fresh run on the full public dataset or for campus field validation.

## 8. Claim boundary

- Report predictive metrics only with dataset, split and sample size.
- Do not equate prediction accuracy with energy savings or causal effects.
- Do not interpret `meter_reading` as kWh or calculate verified carbon reductions until the exact dataset derivative's unit lineage is confirmed.
- Scenario savings and carbon are conditional assumptions, not measured outcomes.
