import pandas as pd
from src.pipeline import features,anomaly_table,baseline_metrics,evaluate_unseen_buildings,train_model

def toy():
    dates=pd.date_range("2017-01-01",periods=70)
    return pd.DataFrame({"building_id":["B"]*70,"date":dates,"meter_reading":[100.0]*69+[300.0],"sub_primaryspaceusage":["Classroom"]*70,"sqm":[1000.0]*70,"site_id":["S1"]*70,"airTemperature":[20.0]*70,"cloudCoverage":[2.0]*70,"dewTemperature":[10.0]*70,"windSpeed":[2.0]*70})

def test_shifted_lag():
    d,f=features(toy()); assert d.loc[d.date.eq(pd.Timestamp("2017-01-29")),"lag1"].iloc[0]==100.0

def test_anomaly_output():
    out=anomaly_table(toy()); assert "priority_score" in out.columns and len(out)==1


def test_median_baseline_does_not_cross_building_boundaries():
    dates = pd.date_range("2026-01-01", periods=10)
    frame = pd.concat([
        pd.DataFrame({"building_id": "A", "date": dates, "meter_reading": 10.0}),
        pd.DataFrame({"building_id": "B", "date": dates, "meter_reading": 1000.0}),
    ], ignore_index=True)
    metrics = baseline_metrics(frame)
    # Stable buildings should not borrow meter history from another building.
    assert metrics["median_28"]["mae"] == 0.0
    assert metrics["median_28"]["wape"] == 0.0


def test_unseen_building_evaluation_keeps_groups_disjoint():
    frames = []
    for idx in range(6):
        frame = toy()
        frame["building_id"] = f"B{idx}"
        frame["meter_reading"] = frame["meter_reading"] + idx * 20
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True).sort_values(["building_id", "date"])
    engineered, feature_cols = features(raw)
    _, train, test, _, metrics = evaluate_unseen_buildings(
        engineered, feature_cols, test_size=0.33, seed=7
    )
    assert set(train["building_id"]).isdisjoint(set(test["building_id"]))
    assert set(metrics["train_buildings"]).isdisjoint(set(metrics["test_buildings"]))
    assert metrics["split"] == "unseen_building_group_holdout"


def test_unseen_building_evaluation_rejects_invalid_test_size():
    raw = pd.concat([toy().assign(building_id="A"), toy().assign(building_id="B")], ignore_index=True)
    engineered, feature_cols = features(raw)
    import pytest
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        evaluate_unseen_buildings(engineered, feature_cols, test_size=1.0)


def test_unseen_building_evaluation_requires_multiple_buildings():
    engineered, feature_cols = features(toy())
    import pytest
    with pytest.raises(ValueError, match="At least two buildings"):
        evaluate_unseen_buildings(engineered, feature_cols)


def test_baselines_are_scored_on_chronological_holdout_rows():
    frames = []
    for building, level in [("A", 100.0), ("B", 250.0)]:
        frame = toy()
        frame["building_id"] = building
        frame["meter_reading"] = level
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    engineered, feature_cols = features(raw)
    from src.pipeline import train_model
    _, _, test, _, _, metrics = train_model(engineered, feature_cols)
    baselines = metrics["baseline_benchmarks"]
    assert metrics["baseline_comparison_rows"] == len(test)
    # Feature rows require 28 days of history, so all three past-only baselines
    # should be available on the same chronological holdout rows.
    assert all(item["n"] == len(test) for item in baselines.values())


def test_prediction_features_do_not_use_current_day_reading():
    raw = toy()
    engineered, feature_cols = features(raw)
    row = engineered.loc[engineered["date"].eq(pd.Timestamp("2017-03-11"))].iloc[0]
    assert row["lag1"] == raw.loc[raw["date"].eq(pd.Timestamp("2017-03-10")), "meter_reading"].iloc[0]
    assert row["lag1"] != row["meter_reading"]


def test_calendar_baselines_do_not_treat_previous_row_as_previous_day():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-09", "2026-01-10"])
    frame = pd.DataFrame({
        "building_id": ["A"] * 4,
        "date": dates,
        "meter_reading": [10.0, 20.0, 70.0, 80.0],
    })
    evaluation = frame.iloc[[2, 3]].copy()
    metrics = baseline_metrics(frame, evaluation_frame=evaluation)
    # Jan 9 has an exact 7-day predecessor (Jan 2), while Jan 10 has
    # no Jan 9 predecessor in this dataset. A row-shift baseline would differ.
    assert metrics["seasonal_7d"]["n"] == 1
    assert metrics["seasonal_7d"]["mae"] == 50.0
    assert metrics["naive_1d"]["n"] == 1
    assert metrics["naive_1d"]["mae"] == 10.0


def test_model_and_baselines_report_fair_common_row_cohort():
    # Missing calendar dates intentionally reduce baseline coverage. The model
    # and all three baselines must also be scored on their shared intersection.
    dates = pd.date_range("2026-01-01", periods=100).delete([35, 42, 57])
    data = pd.DataFrame({
        "building_id": ["A"] * len(dates),
        "date": dates,
        "meter_reading": [100.0 + (i % 7) * 3 + i * 0.1 for i in range(len(dates))],
        "sqm": [500.0] * len(dates),
    })
    engineered, feature_columns = features(data)
    _, _, test_frame, _, _, metrics = train_model(engineered, feature_columns)
    comparison = metrics["common_baseline_comparison"]
    model = comparison["model"]
    baselines = comparison["baselines"]
    assert comparison["evaluation_cohort"].startswith("intersection")
    assert model["n"] > 0
    assert model["n"] <= len(test_frame)
    assert all(item["n"] == model["n"] for item in baselines.values())
    assert 0 < model["coverage_of_chronological_test"] <= 1
    assert set(baselines) == {"naive_1d", "seasonal_7d", "median_28"}


def test_baseline_predictions_are_past_only_when_future_values_change():
    dates = pd.date_range("2026-01-01", periods=45)
    data = pd.DataFrame({
        "building_id": ["A"] * len(dates),
        "date": dates,
        "meter_reading": [float(i + 10) for i in range(len(dates))],
    })
    evaluation = data.iloc[[30, 31, 32]].copy()
    from src.pipeline import baseline_predictions
    first = baseline_predictions(data, evaluation)
    changed = data.copy()
    changed.loc[changed["date"] > pd.Timestamp("2026-02-01"), "meter_reading"] = 99999.0
    second = baseline_predictions(changed, evaluation)
    # Predictions for Feb 1 and earlier must not use observations after target date.
    pd.testing.assert_series_equal(first["naive_1d"], second["naive_1d"])
    pd.testing.assert_series_equal(first["seasonal_7d"], second["seasonal_7d"])
    pd.testing.assert_series_equal(first["median_28"], second["median_28"])
