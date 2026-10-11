import pandas as pd
from src.pipeline import features,anomaly_table,baseline_metrics,evaluate_unseen_buildings

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
