"""Field-validation utilities for preliminary energy-savings evidence."""
import numpy as np
import pandas as pd


def _prepare(frame, intervention_date, require_both_periods=True):
    required = {"date", "energy", "group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Validation data is empty.")

    v = frame.copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v["energy"] = pd.to_numeric(v["energy"], errors="coerce")
    v["group"] = v["group"].astype(str).str.lower().str.strip()
    v = v.dropna(subset=["date", "energy"]).copy()
    v = v[v["group"].isin(["treated", "control"])].copy()
    if v.empty:
        raise ValueError("No valid treated/control observations remain.")
    if (v["energy"] < 0).any():
        raise ValueError("Energy values must be non-negative.")
    if v.duplicated(["building_id", "date"]).any() if "building_id" in v.columns else v.duplicated(["group", "date"]).any():
        raise ValueError("Duplicate building-date (or group-date) observations found; aggregate or deduplicate them first.")

    cutoff = pd.to_datetime(intervention_date, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("A valid intervention date is required.")
    v["period"] = np.where(v["date"] < cutoff, "pre", "post")

    for group in ["treated", "control"]:
        g = v[v["group"] == group]
        if g.empty or (require_both_periods and set(g["period"]) != {"pre", "post"}):
            raise ValueError(
                f"Both pre- and post-intervention observations are required for the {group} group."
            )
    return v


def did_effect(frame, intervention_date):
    v = _prepare(frame, intervention_date)
    summary = v.groupby(["group", "period"])["energy"].mean().unstack()
    tc = float(summary.loc["treated", "post"] - summary.loc["treated", "pre"])
    cc = float(summary.loc["control", "post"] - summary.loc["control", "pre"])
    effect = tc - cc
    base = float(summary.loc["treated", "pre"])
    return {
        "treated_change": tc,
        "control_change": cc,
        "did_abs": effect,
        "did_pct": effect / base * 100 if base else float("nan"),
        "summary": summary,
    }


def pretrend_check(frame, intervention_date):
    v = _prepare(frame, intervention_date, require_both_periods=False)
    pre = v[v["period"] == "pre"].copy()
    if pre["date"].nunique() < 3:
        return {"status": "insufficient", "treated_slope": np.nan, "control_slope": np.nan, "slope_gap": np.nan}
    pre["t"] = (pre["date"] - pre["date"].min()).dt.days.astype(float)
    slopes = {}
    for group in ["treated", "control"]:
        g = pre[pre["group"] == group]
        slopes[group] = float(np.polyfit(g["t"], g["energy"], 1)[0]) if len(g) >= 3 and g["t"].nunique() >= 2 else np.nan
    if np.isnan(slopes["treated"]) or np.isnan(slopes["control"]):
        return {"status": "insufficient", "treated_slope": slopes["treated"], "control_slope": slopes["control"], "slope_gap": np.nan}
    scale = max(abs(slopes["treated"]) + abs(slopes["control"]), 1e-9)
    return {
        "status": "diagnostic",
        "treated_slope": slopes["treated"],
        "control_slope": slopes["control"],
        "slope_gap": slopes["treated"] - slopes["control"],
        "relative_gap": abs(slopes["treated"] - slopes["control"]) / scale,
    }


def placebo_effect(frame, placebo_date):
    return did_effect(frame, placebo_date)


def bootstrap_did(frame, intervention_date, n_boot=300, seed=42):
    if "building_id" not in frame.columns:
        return {"status": "insufficient", "n": 0, "reason": "building_id is required for clustered bootstrap."}
    v = _prepare(frame, intervention_date)
    ids = v["building_id"].dropna().unique()
    if len(ids) < 2:
        return {"status": "insufficient", "n": 0}
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1.")
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        parts = []
        for index, bid in enumerate(sampled):
            part = v[v["building_id"] == bid].copy()
            # Give repeated bootstrap draws unique IDs so duplicate sampled clusters
            # remain separate clusters and do not trigger duplicate-key validation.
            part["building_id"] = f"bootstrap_{index}"
            parts.append(part)
        b = pd.concat(parts, ignore_index=True)
        try:
            effects.append(did_effect(b, intervention_date)["did_abs"])
        except ValueError:
            continue
    if not effects:
        return {"status": "insufficient", "n": 0}
    arr = np.asarray(effects)
    return {
        "status": "ok",
        "n": len(arr),
        "estimate": float(did_effect(v, intervention_date)["did_abs"]),
        "ci_low": float(np.quantile(arr, .025)),
        "ci_high": float(np.quantile(arr, .975)),
        "share_negative": float((arr < 0).mean()),
    }
