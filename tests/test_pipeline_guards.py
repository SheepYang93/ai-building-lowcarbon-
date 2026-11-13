import pandas as pd
import pytest

from src.pipeline import REQUIRED, anomaly_table, baseline_metrics, baseline_predictions, carbon_scenario, features, load_filter, train_model


def test_load_filter_reports_missing_required_columns(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"date": ["2026-01-01"], "meter_reading": [1.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_filter(path)


def test_train_model_rejects_empty_complete_feature_rows():
    columns = ["date", "meter_reading", "building_id", "missing_feature"]
    data = pd.DataFrame(columns=columns)

    with pytest.raises(ValueError, match="No complete rows remain"):
        train_model(data, ["missing_feature"])


def test_train_model_rejects_single_date():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "meter_reading": [10.0, 11.0],
        "feature": [1.0, 2.0],
    })

    with pytest.raises(ValueError, match="At least two distinct dates"):
        train_model(data, ["feature"])


def test_features_work_without_optional_weather_columns():
    data = pd.DataFrame({
        "building_id": ["b1"] * 40,
        "date": pd.date_range("2026-01-01", periods=40),
        "meter_reading": [100.0] * 40,
        "sqm": [500.0] * 40,
    })

    engineered, feature_columns = features(data)

    assert "sqm" in feature_columns
    assert "airTemperature" not in feature_columns
    assert "lag28" in engineered.columns


def test_features_sort_each_building_history_before_lags():
    data = pd.DataFrame({
        "building_id": ["b1"] * 40,
        "date": list(pd.date_range("2026-01-01", periods=40)[::-1]),
        "meter_reading": list(range(40)),
        "sqm": [500.0] * 40,
    })

    engineered, _ = features(data)

    assert engineered["date"].is_monotonic_increasing
    assert engineered.loc[28, "lag1"] == engineered.loc[27, "meter_reading"]


def test_partial_weather_missingness_does_not_impute_or_drop_rows():
    data = pd.DataFrame({
        "building_id": ["b1"] * 40,
        "date": pd.date_range("2026-01-01", periods=40),
        "meter_reading": [100.0] * 40,
        "sqm": [500.0] * 40,
        "airTemperature": [10.0] * 39 + [float("nan")],
    })

    engineered, feature_columns = features(data)

    assert len(engineered) == 40
    assert "airTemperature" not in feature_columns
    assert pd.isna(engineered.loc[39, "airTemperature"])


def test_anomaly_table_accepts_missing_optional_site_id():
    data = pd.DataFrame({
        "building_id": ["b1"] * 70,
        "date": pd.date_range("2026-01-01", periods=70),
        "meter_reading": [100.0] * 69 + [150.0],
        "sub_primaryspaceusage": ["Education"] * 70,
        "sqm": [500.0] * 70,
    })

    result = anomaly_table(data)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "site_id"])



def test_baselines_use_exact_model_holdout_rows():
    data = pd.DataFrame({
        "building_id": ["b1"] * 80,
        "date": pd.date_range("2026-01-01", periods=80),
        "meter_reading": [100.0 + (i % 7) * 5 + i * 0.2 for i in range(80)],
        "sqm": [500.0] * 80,
    })
    engineered, feature_columns = features(data)
    _, _, test_frame, _, _, metrics = train_model(engineered, feature_columns)

    baselines = metrics["baseline_benchmarks"]

    assert set(baselines) == {"naive_1d", "seasonal_7d", "median_28"}
    assert all(item["n"] == len(test_frame) for item in baselines.values())
    assert all(item["mae"] >= 0 and item["rmse"] >= 0 and item["wape"] >= 0 for item in baselines.values())


def test_carbon_scenario_labels_observed_period_not_annual():
    data = pd.DataFrame({
        "building_id": ["b1"] * 30,
        "meter_reading": [100.0] * 30,
    })

    result = carbon_scenario(data, reduction=0.2, ef=0.5)

    assert result.loc[0, "period_meter_reading"] == 3000.0
    assert "annual_meter_reading" not in result.columns
    assert result.loc[0, "scenario_reduction"] == 600.0


def test_baselines_are_invariant_to_future_meter_reading_changes():
    dates = pd.date_range("2026-01-01", periods=60)
    history = pd.DataFrame({
        "building_id": ["b1"] * 60,
        "date": dates,
        "meter_reading": [100.0 + i for i in range(60)],
    })
    evaluation = history.loc[history["date"] == dates[45]].copy()
    original = baseline_predictions(history, evaluation)

    changed = history.copy()
    changed.loc[changed["date"] > dates[45], "meter_reading"] += 10000
    perturbed = baseline_predictions(changed, evaluation)

    for column in ["naive_1d", "seasonal_7d", "median_28"]:
        pd.testing.assert_series_equal(original[column], perturbed[column])


def test_building_group_holdout_keeps_buildings_disjoint_and_reports_history_boundary():
    from src.pipeline import evaluate_unseen_buildings

    rows = []
    for building_idx in range(4):
        building_id = f"b{building_idx}"
        for day, date in enumerate(pd.date_range("2026-01-01", periods=40)):
            rows.append({
                "building_id": building_id,
                "date": date,
                "meter_reading": 100.0 + building_idx * 10 + day % 7,
                "sqm": 500.0 + building_idx,
                "feature": float(day % 7),
            })
    data = pd.DataFrame(rows)

    _, train, test, pred, metrics = evaluate_unseen_buildings(
        data, ["feature", "sqm"], test_size=0.5, seed=17
    )

    train_buildings = set(train["building_id"])
    test_buildings = set(test["building_id"])
    assert train_buildings.isdisjoint(test_buildings)
    assert set(metrics["train_buildings"]).isdisjoint(metrics["test_buildings"])
    assert len(pred) == len(test)
    assert metrics["split"] == "unseen_building_group_holdout"
    assert "past-only lag features" in metrics["history_note"]


def test_regression_metrics_mark_undefined_wape_and_single_sample_r2():
    import numpy as np
    from src.pipeline import _regression_metrics

    metrics = _regression_metrics([0.0], [2.0])
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert np.isnan(metrics["wape"])
    assert metrics["wape_status"] == "undefined_zero_actual_sum"
    assert np.isnan(metrics["r2"])
    assert metrics["r2_status"] == "undefined_single_sample"


def test_regression_metrics_reject_nonfinite_inputs():
    from src.pipeline import _regression_metrics

    with pytest.raises(ValueError, match="finite"):
        _regression_metrics([1.0, float("nan")], [1.0, 2.0])


def test_regression_metrics_zero_actual_and_single_row_are_explicit():
    import numpy as np
    from src.pipeline import _regression_metrics
    result = _regression_metrics([0.0], [1.5])
    assert result["mae"] == 1.5
    assert result["rmse"] == 1.5
    assert np.isnan(result["wape"]) and result["wape_status"] == "undefined_zero_actual_sum"
    assert np.isnan(result["r2"]) and result["r2_status"] == "undefined_single_sample"


def test_json_safe_converts_nonfinite_metrics_to_null():
    import numpy as np
    from src.pipeline import json_safe
    safe = json_safe({"wape": float("nan"), "r2": np.float64("inf"), "n": np.int64(2)})
    assert safe == {"wape": None, "r2": None, "n": 2}
