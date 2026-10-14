import pandas as pd
import pytest

from src.pipeline import REQUIRED, anomaly_table, features, load_filter, train_model


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


def test_features_impute_partial_weather_gaps():
    data = pd.DataFrame({
        "building_id": ["b1"] * 40,
        "date": pd.date_range("2026-01-01", periods=40),
        "meter_reading": [100.0 + i for i in range(40)],
        "sqm": [500.0] * 40,
        "airTemperature": [10.0] * 39 + [None],
    })

    engineered, feature_columns = features(data)

    assert "airTemperature" in feature_columns
    assert engineered["airTemperature"].isna().sum() == 0
    assert engineered["airTemperature"].iloc[-1] == 10.0


def test_features_impute_partial_weather_gaps_without_dropping_rows():
    data = pd.DataFrame({
        "building_id": ["b1"] * 40,
        "date": pd.date_range("2026-01-01", periods=40),
        "meter_reading": [100.0 + i for i in range(40)],
        "sqm": [500.0] * 40,
        "airTemperature": [10.0, None, 12.0, None] + [14.0] * 36,
    })

    engineered, feature_columns = features(data)

    assert "airTemperature" in feature_columns
    assert engineered["airTemperature"].isna().sum() == 0
    assert len(engineered) == len(data)


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
