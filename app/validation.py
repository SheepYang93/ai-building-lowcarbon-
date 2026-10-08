"""Field-validation utilities for preliminary energy-savings evidence."""
import numpy as np
import pandas as pd


def did_effect(frame, intervention_date):
    v = frame.copy()
    v["period"] = v["date"].lt(pd.Timestamp(intervention_date)).map({True: "pre", False: "post"})
    summary = v.groupby(["group", "period"])["energy"].mean().unstack()
    if not {"treated", "control"}.issubset(summary.index) or not {"pre", "post"}.issubset(summary.columns):
        raise ValueError("Both treated/control groups and pre/post periods are required.")
    treated_change = float(summary.loc["treated", "post"] - summary.loc["treated", "pre"])
    control_change = float(summary.loc["control", "post"] - summary.loc["control", "pre"])
    effect = treated_change - control_change
    baseline = float(summary.loc["treated", "pre"])
    relative = effect / baseline * 100 if baseline else float("nan")
    return {
        "treated_change": treated_change,
        "control_change": control_change,
        "did_abs": effect,
        "did_pct": relative,
        "summary": summary,
    }


def pretrend_check(frame, intervention_date):
    """Compare simple pre-period linear slopes; diagnostic, not a formal test."""
    v = frame.copy()
    cutoff = pd.Timestamp(intervention_date)
    pre = v[v["date"] < cutoff].copy()
    if pre["date"].nunique() < 3:
        return {"status": "insufficient", "treated_slope": np.nan, "control_slope": np.nan, "slope_gap": np.nan}
    pre["t"] = (pre["date"] - pre["date"].min()).dt.days.astype(float)
    slopes = {}
    for group in ["treated", "control"]:
        g = pre[pre["group"] == group].dropna(subset=["t", "energy"])
        if len(g) < 3 or g["t"].nunique() < 2:
            slopes[group] = np.nan
        else:
            slopes[group] = float(np.polyfit(g["t"], g["energy"], 1)[0])
    if np.isnan(slopes["treated"]) or np.isnan(slopes["control"]):
        return {"status": "insufficient", "treated_slope": slopes["treated"], "control_slope": slopes["control"], "slope_gap": np.nan}
    return {
        "status": "diagnostic",
        "treated_slope": slopes["treated"],
        "control_slope": slopes["control"],
        "slope_gap": slopes["treated"] - slopes["control"],
    }
