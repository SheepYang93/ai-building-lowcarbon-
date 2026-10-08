"""UI-independent robust building-energy diagnostics."""

import numpy as np
import pandas as pd

from app.config import CONFIG, evidence_level


def diagnose_uploaded_data(daily):
    d = daily.copy()
    d["weekday"] = d["date"].dt.dayofweek < 5
    d["baseline_28d"] = d["energy"].shift(1).rolling(CONFIG.baseline_window, min_periods=CONFIG.baseline_min_periods).median()
    d["mad_28d"] = d["energy"].shift(1).rolling(CONFIG.mad_window, min_periods=CONFIG.mad_min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    scale = (1.4826 * d["mad_28d"]).clip(lower=1e-9)
    d["robust_z"] = (d["energy"] - d["baseline_28d"]) / scale
    d["ratio_to_baseline"] = d["energy"] / d["baseline_28d"].replace(0, np.nan)
    d["anomaly"] = (
        d["baseline_28d"].notna()
        & ((d["robust_z"] >= CONFIG.robust_z_threshold) & (d["ratio_to_baseline"] >= 1 + CONFIG.relative_gap_threshold))
    )

    valid = d[d["baseline_28d"].notna()].copy()
    anomaly_days = int(valid["anomaly"].sum())
    anomaly_share = float(valid["anomaly"].mean()) if len(valid) else 0.0
    total_energy = float(d["energy"].sum())
    baseline_energy = float(d["baseline_28d"].fillna(d["energy"]).sum())
    potential_pct = (
        max(0.0, (total_energy - baseline_energy) / total_energy * 100)
        if total_energy > 0
        else 0.0
    )

    weekday = d.loc[d["weekday"], "energy"].median()
    weekend = d.loc[~d["weekday"], "energy"].median()
    weekend_ratio = weekend / weekday if weekday and not pd.isna(weekday) else np.nan

    recent = d.tail(min(28, len(d)))
    early = d.head(min(28, len(d)))
    trend_pct = (
        (recent["energy"].median() / early["energy"].median() - 1) * 100
        if early["energy"].median() > 0
        else 0.0
    )

    if anomaly_days == 0:
        fingerprint = "stable / no robust anomaly"
        intervention = "normal monitoring"
    elif trend_pct >= 15:
        fingerprint = "gradual operational drift"
        intervention = "maintenance + HVAC inspection"
    elif anomaly_share >= 0.20:
        fingerprint = "persistent high-use"
        intervention = "HVAC + schedule control"
    elif weekend_ratio >= 0.75:
        fingerprint = "recurrent / schedule-related"
        intervention = "schedule + occupancy control"
    else:
        fingerprint = "intermittent operational anomaly"
        intervention = "schedule + equipment check"

    return d, {
        "records": len(d),
        "start": d["date"].min().date().isoformat(),
        "end": d["date"].max().date().isoformat(),
        "total_energy": total_energy,
        "anomaly_days": anomaly_days,
        "anomaly_share": anomaly_share,
        "potential_pct": potential_pct,
        "weekend_ratio": weekend_ratio,
        "trend_pct": trend_pct,
        "fingerprint": fingerprint,
        "intervention": intervention,
        "evidence_level": evidence_level(100.0 if len(d) >= 84 else 75.0 if len(d) >= 56 else 55.0),
    }

