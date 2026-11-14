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


def test_baseline_metrics_use_undefined_wape_and_report_coverage():
    import numpy as np
    import pandas as pd
    from src.pipeline import baseline_metrics

    history = pd.DataFrame({
        "building_id": ["B1"] * 12,
        "date": pd.date_range("2024-01-01", periods=12, freq="D"),
        "meter_reading": [0.0] * 12,
    })
    evaluation = history.iloc[-3:].copy()
    scores = baseline_metrics(history, evaluation_frame=evaluation)
    assert set(scores) == {"naive_1d", "seasonal_7d", "median_28"}
    for score in scores.values():
        assert score["n"] == score["valid_rows"] == 3
        assert score["total_rows"] == 3 and score["coverage"] == 1.0
        assert np.isnan(score["wape"])
        assert score["wape_status"] == "undefined_zero_actual_sum"


def test_train_model_survives_empty_fair_comparison_cohort(monkeypatch):
    import numpy as np
    import pandas as pd
    import src.pipeline as pipeline

    frame = pd.DataFrame({
        "building_id": ["B1"] * 20,
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "meter_reading": np.arange(20, dtype=float) + 10,
        "feature": np.arange(20, dtype=float),
    })
    def no_baseline_rows(history, evaluation):
        result = evaluation[["building_id", "date", "meter_reading"]].copy()
        for name in ["naive_1d", "seasonal_7d", "median_28"]:
            result[name] = np.nan
        return result
    monkeypatch.setattr(pipeline, "baseline_predictions", no_baseline_rows)
    _, _, _, _, _, metrics = pipeline.train_model(frame, ["feature"], cap=100)
    fair = metrics["common_baseline_comparison"]
    assert fair["model"]["n"] == 0
    assert fair["model"]["status"] == "no_shared_valid_rows"
    assert fair["model"]["coverage_of_chronological_test"] == 0
    assert all(item["status"] == "no_shared_valid_rows" for item in fair["baselines"].values())


def test_baseline_metrics_returns_explicit_status_when_no_prediction_is_valid():
    import numpy as np
    import pandas as pd
    from src.pipeline import baseline_metrics

    one = pd.DataFrame({
        "building_id": ["B1"], "date": pd.to_datetime(["2024-01-01"]),
        "meter_reading": [10.0],
    })
    scores = baseline_metrics(one, evaluation_frame=one)
    for score in scores.values():
        assert score["n"] == 0
        assert score["coverage"] == 0.0
        assert score["status"] == "no_valid_predictions"
        assert np.isnan(score["mae"]) and np.isnan(score["wape"])


def test_training_report_script_compiles_without_execution():
    from pathlib import Path
    source = Path("scripts/run_v3.py").read_text(encoding="utf-8")
    compile(source, "scripts/run_v3.py", "exec")
    assert 'markdown_to_docx(report_path, out/"evaluation_report.docx")' in source


def test_markdown_report_exports_valid_word_with_headings_tables_and_bullets(tmp_path):
    from docx import Document
    from src.reporting import markdown_to_docx

    markdown = tmp_path / "report.md"
    output = tmp_path / "nested" / "report.docx"
    markdown.write_text(
        """# Evaluation

## Metrics
- Rows: 12

| Metric | Value |
|---|---:|
| MAE | 2.5 |
| RMSE | 3.1 |
""",
        encoding="utf-8",
    )

    result = markdown_to_docx(markdown, output)
    document = Document(result)

    assert output.exists()
    assert document.core_properties.title == "Building Energy Model Evaluation"
    assert any(p.text == "Evaluation" for p in document.paragraphs)
    assert any(p.text == "Metrics" for p in document.paragraphs)
    assert any(p.text == "Rows: 12" for p in document.paragraphs)
    assert len(document.tables) == 1
    assert [cell.text for cell in document.tables[0].rows[0].cells] == ["Metric", "Value"]
    assert [cell.text for cell in document.tables[0].rows[1].cells] == ["MAE", "2.5"]


def test_building_error_summary_handles_bias_zero_actual_and_ape_coverage():
    import numpy as np
    import pandas as pd
    from src.reporting import building_error_summary

    rows = pd.DataFrame({
        "building_id": ["A", "A", "A", "B", "B"],
        "actual": [10.0, 0.0, 20.0, 0.0, 0.0],
        "predicted": [8.0, 3.0, 25.0, 1.0, -1.0],
    })
    summary = building_error_summary(rows).set_index("building_id")

    assert summary.loc["A", "test_rows"] == 3
    assert summary.loc["A", "mae"] == 10 / 3
    assert summary.loc["A", "mean_error"] == -2
    assert summary.loc["A", "bias_direction"] == "overprediction"
    assert summary.loc["A", "ape_valid_rows"] == 2
    assert summary.loc["A", "median_absolute_percentage_error"] == 0.225
    assert summary.loc["B", "wape"] != summary.loc["B", "wape"]  # undefined: zero actual sum
    assert summary.loc["B", "ape_valid_rows"] == 0


def test_building_error_summary_rejects_missing_columns_and_empty_rows():
    import pandas as pd
    import pytest
    from src.reporting import building_error_summary

    with pytest.raises(ValueError, match="missing columns"):
        building_error_summary(pd.DataFrame({"actual": [1], "predicted": [1]}))
    result = building_error_summary(pd.DataFrame({
        "building_id": ["A"], "actual": [float("nan")], "predicted": [1.0]
    }))
    assert result.empty



def test_word_report_table_header_is_bold_and_ragged_rows_are_padded(tmp_path):
    from docx import Document
    from src.reporting import markdown_to_docx

    markdown = tmp_path / "table.md"
    output = tmp_path / "table.docx"
    markdown.write_text(
        "| Metric | Value | Notes |\n|---|---:|---|\n| MAE | 2.1 |\n",
        encoding="utf-8",
    )
    markdown_to_docx(markdown, output)
    table = Document(output).tables[0]

    assert len(table.columns) == 3
    assert [cell.text for cell in table.rows[0].cells] == ["Metric", "Value", "Notes"]
    assert all(run.bold for cell in table.rows[0].cells for paragraph in cell.paragraphs for run in paragraph.runs)
    assert [cell.text for cell in table.rows[1].cells] == ["MAE", "2.1", ""]


def test_word_report_ignores_markdown_table_separator_without_header(tmp_path):
    from docx import Document
    from src.reporting import markdown_to_docx

    markdown = tmp_path / "separator.md"
    output = tmp_path / "separator.docx"
    markdown.write_text("|---|---:|\n", encoding="utf-8")
    markdown_to_docx(markdown, output)

    assert Document(output).tables == []


def test_building_error_summary_sorts_highest_mae_and_filters_nonfinite():
    import pandas as pd
    from src.reporting import building_error_summary

    frame = pd.DataFrame({
        "building_id": ["low", "high", "high", "invalid"],
        "actual": [10.0, 10.0, 20.0, float("inf")],
        "predicted": [9.0, 0.0, 0.0, 1.0],
    })
    result = building_error_summary(frame)

    assert result["building_id"].tolist() == ["high", "low"]
    assert result.iloc[0]["mae"] == 15.0
    assert result.iloc[0]["test_rows"] == 2
    assert "invalid" not in result["building_id"].tolist()


def test_streamlit_demo_and_training_script_compile():
    from pathlib import Path

    for path in ["app/streamlit_app.py", "scripts/run_v3.py", "src/reporting.py"]:
        source = Path(path).read_text(encoding="utf-8")
        compile(source, path, "exec")


def test_word_export_handles_empty_report_and_blank_lines(tmp_path):
    from docx import Document
    from src.reporting import markdown_to_docx

    markdown = tmp_path / "empty.md"
    output = tmp_path / "empty.docx"
    markdown.write_text("\n\n", encoding="utf-8")
    markdown_to_docx(markdown, output)
    document = Document(output)
    assert output.exists()
    assert len(document.tables) == 0
    assert len(document.paragraphs) == 0


def test_metrics_report_formatter_uses_na_for_nonfinite_values():
    import numpy as np
    from src.pipeline import json_safe

    safe = json_safe({"mae": np.nan, "rmse": np.inf, "wape": -np.inf})
    assert safe == {"mae": None, "rmse": None, "wape": None}


def test_dashboard_multiline_translations_are_interpolated():
    from pathlib import Path

    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert '            f\'\'\'\n<div class="insight-card">' in source
    assert '            f\'\'\'\n<span class="chip">' in source
    assert '            f\'\'\'\n**{tr("能源数据"' in source



def test_upload_quality_score_does_not_penalize_optional_weather():
    import ast
    from pathlib import Path
    import pandas as pd

    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "score_upload_quality"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), "<score_upload_quality>", "exec"), namespace)
    score_upload_quality = namespace["score_upload_quality"]

    energy_only = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10, freq="D"),
        "energy": [10.0] * 10,
    })
    with_weather = energy_only.assign(outdoor_temperature=[20.0] * 10)
    score_without_weather, checks_without_weather = score_upload_quality(energy_only)
    score_with_weather, checks_with_weather = score_upload_quality(with_weather)

    assert score_without_weather == 100.0
    assert score_with_weather == 100.0
    assert checks_without_weather["temperature_available"] == 0.0
    assert checks_with_weather["temperature_available"] == 1.0
    assert checks_without_weather["weather_optional"] is True

    # Missing calendar days lower the core score through temporal coverage.
    gapped = energy_only.iloc[[0, 1, 2, 4, 5, 6, 7, 9]].copy()
    gap_score, gap_checks = score_upload_quality(gapped)
    assert gap_checks["temporal_coverage"] == 0.8
    assert gap_score < 100.0

    # Duplicate raw meter timestamps are visible in the core score.
    raw_duplicate = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 00:00",
            "2026-01-02 00:00", "2026-01-03 00:00",
        ])
    })
    duplicate_score, duplicate_checks = score_upload_quality(energy_only, raw_duplicate)
    assert duplicate_checks["no_duplicate_timestamps"] == 0.75
    assert duplicate_score < 100.0


def test_building_error_summary_zero_denominators_do_not_warn_for_ape_or_wape():
    import warnings
    import pandas as pd
    from src.reporting import building_error_summary

    frame = pd.DataFrame({
        "building_id": ["zero", "zero", "positive"],
        "actual": [0.0, 0.0, 10.0],
        "predicted": [2.0, -1.0, 8.0],
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = building_error_summary(frame).set_index("building_id")

    assert summary.loc["zero", "ape_valid_rows"] == 0
    assert pd.isna(summary.loc["zero", "median_absolute_percentage_error"])
    assert pd.isna(summary.loc["zero", "wape"])
    assert summary.loc["positive", "ape_valid_rows"] == 1
    assert not any("divide by zero" in str(item.message).lower() for item in caught)


def test_run_v3_rejects_invalid_experiment_parameters_before_data_access():
    import subprocess
    import sys

    script = "scripts/run_v3.py"
    for args, expected in [
        (["--input", "missing.csv", "--train-cap", "0"], "--train-cap must be a positive integer"),
        (["--input", "missing.csv", "--building-test-size", "1"], "--building-test-size must be strictly between 0 and 1"),
    ]:
        result = subprocess.run(
            [sys.executable, script, *args], capture_output=True, text=True, check=False
        )
        assert result.returncode != 0
        assert expected in result.stderr
        assert "Input CSV not found" not in result.stderr


def test_chronological_export_ape_uses_masked_division_for_zero_actuals():
    from pathlib import Path

    source = Path("scripts/run_v3.py").read_text(encoding="utf-8")
    assert 'chronological_predictions["absolute_percentage_error"] = np.nan' in source
    assert 'valid_ape = actual_abs > 0' in source
    assert 'chronological_predictions.loc[valid_ape, "absolute_error"] / actual_abs.loc[valid_ape]' in source
    assert 'np.where(\n    actual_abs > 0' not in source


def test_load_filter_rejects_duplicate_building_date_daily_readings(tmp_path):
    import pandas as pd
    import pytest
    from src.pipeline import load_filter

    frame = pd.DataFrame({
        "building_id": ["B1", "B1"],
        "meter": ["electricity", "electricity"],
        "date": ["2024-01-01", "2024-01-01"],
        "meter_reading": [10.0, 11.0],
        "sub_primaryspaceusage": ["Education", "Education"],
        "sqm": [100.0, 100.0],
    })
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate building/date readings"):
        load_filter(path)
