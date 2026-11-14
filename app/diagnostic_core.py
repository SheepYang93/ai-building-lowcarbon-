"""UI-independent robust building-energy diagnostics."""

import numpy as np
import pandas as pd

try:
    from app.config import CONFIG, evidence_level
except ImportError:
    from config import CONFIG, evidence_level


def _robust_mad(values):
    """Return MAD while ignoring the leading NaN introduced by shift(1)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def diagnose_uploaded_data(daily):
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d["energy"] = pd.to_numeric(d["energy"], errors="coerce")
    d = d.dropna(subset=["date", "energy"])
    d = d[d["energy"] >= 0].sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("Diagnostic input has no valid non-negative energy observations.")
    if d["date"].duplicated().any():
        raise ValueError("Diagnostic input must contain at most one row per date.")

    # Use calendar days rather than row counts so missing dates do not compress
    # the reference windows and accidentally pull older readings into baseline.
    calendar = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    energy = d.set_index("date")["energy"].reindex(calendar)
    baseline = energy.shift(1).rolling(
        CONFIG.baseline_window,
        min_periods=CONFIG.baseline_min_periods,
    ).median()
    residual = energy - baseline
    mad = residual.shift(1).rolling(
        CONFIG.mad_window,
        min_periods=CONFIG.mad_min_periods,
    ).apply(_robust_mad, raw=True)
    d = d.set_index("date")
    d["baseline_reference"] = baseline.reindex(d.index).to_numpy()
    d["mad_residual"] = mad.reindex(d.index).to_numpy()
    d = d.reset_index()
    d["weekday"] = d["date"].dt.day_of_week < 5

    # A zero MAD means the recent history is flat. In that case a sufficiently
    # large relative jump is itself strong evidence of an operational anomaly.
    scale = 1.4826 * d["mad_residual"]
    d["robust_z"] = np.where(
        scale > 1e-9,
        (d["energy"] - d["baseline_reference"]) / scale,
        np.where(
            (d["baseline_reference"] > 0)
            & (d["energy"] > d["baseline_reference"]),
            np.inf,
            0.0,
        ),
    )
    d["ratio_to_baseline"] = d["energy"] / d["baseline_reference"].replace(0, np.nan)
    d["anomaly"] = (
        d["baseline_reference"].notna()
        & (
            (d["robust_z"] >= CONFIG.robust_z_threshold)
            & (d["ratio_to_baseline"] >= 1 + CONFIG.relative_gap_threshold)
        )
    )

    valid = d[d["baseline_reference"].notna()].copy()
    anomaly_days = int(valid["anomaly"].sum())
    anomaly_share = float(valid["anomaly"].mean()) if len(valid) else 0.0
    total_energy = float(d["energy"].sum())
    baseline_energy = float(d["baseline_reference"].fillna(d["energy"]).sum())
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
        "evidence_level": evidence_level(
            100.0 if len(d) >= 84 else 75.0 if len(d) >= 56 else 55.0
        ),
    }
