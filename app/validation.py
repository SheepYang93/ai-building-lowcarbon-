"""Simple Difference-in-Differences calculation for field validation."""
import pandas as pd

def did_effect(frame, intervention_date):
    v=frame.copy()
    v["period"]=v["date"].lt(pd.Timestamp(intervention_date)).map({True:"pre",False:"post"})
    summary=v.groupby(["group","period"])["energy"].mean().unstack()
    if not {"treated","control"}.issubset(summary.index) or not {"pre","post"}.issubset(summary.columns):
        raise ValueError("Both treated/control groups and pre/post periods are required.")
    treated_change=float(summary.loc["treated","post"]-summary.loc["treated","pre"])
    control_change=float(summary.loc["control","post"]-summary.loc["control","pre"])
    effect=treated_change-control_change
    baseline=float(summary.loc["treated","pre"])
    relative=effect/baseline*100 if baseline else float("nan")
    return {"treated_change":treated_change,"control_change":control_change,"did_abs":effect,"did_pct":relative,"summary":summary}
