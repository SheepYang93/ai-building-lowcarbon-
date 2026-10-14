import pandas as pd
import pytest

from src.pipeline import REQUIRED, features, load_filter, train_model


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
