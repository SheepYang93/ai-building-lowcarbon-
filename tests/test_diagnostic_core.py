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
