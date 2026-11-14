import numpy as np
import pandas as pd

from app.diagnostic_core import diagnose_uploaded_data


def make_daily(values):
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=len(values)), "energy": values})


def test_diagnostic_uses_past_only_baseline_and_flags_spike():
    diagnosed, summary = diagnose_uploaded_data(make_daily([100.0] * 28 + [130.0, 100.0, 100.0]))
    assert pd.isna(diagnosed.loc[0, "baseline_28d"])
    assert diagnosed.loc[28, "baseline_28d"] == 100.0
    assert bool(diagnosed.loc[28, "anomaly"])
    assert summary["anomaly_days"] >= 1


def test_diagnostic_returns_stable_fingerprint_for_flat_series():
    _, summary = diagnose_uploaded_data(make_daily([100.0] * 40))
    assert summary["anomaly_days"] == 0
    assert summary["fingerprint"] == "stable / no robust anomaly"
    assert np.isclose(summary["potential_pct"], 0.0)


def test_diagnostic_windows_use_calendar_days_when_dates_are_missing():
    dates = list(pd.date_range("2026-01-01", periods=14))
    dates += [pd.Timestamp("2026-01-29"), pd.Timestamp("2026-01-30")]
    frame = pd.DataFrame({
        "date": dates,
        "energy": [100.0] * 14 + [100.0, 150.0],
    })

    diagnosed, _ = diagnose_uploaded_data(frame)

    # Only 13 observed readings fall in the 28 calendar days before Jan 30;
    # a row-count window would incorrectly treat the baseline as complete.
    assert pd.isna(diagnosed.loc[diagnosed["date"] == pd.Timestamp("2026-01-30"), "baseline_28d"].iloc[0])


def test_diagnostic_rejects_duplicate_daily_rows():
    frame = make_daily([100.0, 110.0])
    frame.loc[1, "date"] = frame.loc[0, "date"]

    import pytest
    with pytest.raises(ValueError, match="at most one row per date"):
        diagnose_uploaded_data(frame)
