"""Field-validation utilities for preliminary energy-savings evidence."""
import numpy as np
import pandas as pd

def _prepare(frame, intervention_date):
    v=frame.copy()
    v["date"]=pd.to_datetime(v["date"])
    v["energy"]=pd.to_numeric(v["energy"],errors="coerce")
    v["group"]=v["group"].astype(str).str.lower().str.strip()
    cutoff=pd.Timestamp(intervention_date)
    v["period"]=np.where(v["date"]<cutoff,"pre","post")
    return v.dropna(subset=["date","energy"])

def did_effect(frame, intervention_date):
    v=_prepare(frame,intervention_date)
    summary=v.groupby(["group","period"])["energy"].mean().unstack()
    if not {"treated","control"}.issubset(summary.index) or not {"pre","post"}.issubset(summary.columns):
        raise ValueError("Both treated/control groups and pre/post periods are required.")
    tc=float(summary.loc["treated","post"]-summary.loc["treated","pre"])
    cc=float(summary.loc["control","post"]-summary.loc["control","pre"])
    effect=tc-cc
    base=float(summary.loc["treated","pre"])
    return {"treated_change":tc,"control_change":cc,"did_abs":effect,"did_pct":effect/base*100 if base else float("nan"),"summary":summary}

def pretrend_check(frame, intervention_date):
    v=_prepare(frame,intervention_date)
    pre=v[v["period"]=="pre"].copy()
    if pre["date"].nunique()<3:
        return {"status":"insufficient","treated_slope":np.nan,"control_slope":np.nan,"slope_gap":np.nan}
    pre["t"]=(pre["date"]-pre["date"].min()).dt.days.astype(float)
    slopes={}
    for group in ["treated","control"]:
        g=pre[pre["group"]==group]
        slopes[group]=float(np.polyfit(g["t"],g["energy"],1)[0]) if len(g)>=3 and g["t"].nunique()>=2 else np.nan
    if np.isnan(slopes["treated"]) or np.isnan(slopes["control"]):
        return {"status":"insufficient","treated_slope":slopes["treated"],"control_slope":slopes["control"],"slope_gap":np.nan}
    scale=max(abs(slopes["treated"])+abs(slopes["control"]),1e-9)
    return {"status":"diagnostic","treated_slope":slopes["treated"],"control_slope":slopes["control"],"slope_gap":slopes["treated"]-slopes["control"],"relative_gap":abs(slopes["treated"]-slopes["control"])/scale}

def placebo_effect(frame, placebo_date):
    return did_effect(frame,placebo_date)

def bootstrap_did(frame, intervention_date, n_boot=300, seed=42):
    v=_prepare(frame,intervention_date)
    ids=v["building_id"].dropna().unique()
    if len(ids)<2: return {"status":"insufficient","n":0}
    rng=np.random.default_rng(seed)
    effects=[]
    for _ in range(int(n_boot)):
        sampled=rng.choice(ids,size=len(ids),replace=True)
        b=pd.concat([v[v["building_id"]==bid] for bid in sampled],ignore_index=True)
        try: effects.append(did_effect(b,intervention_date)["did_abs"])
        except ValueError: pass
    if not effects: return {"status":"insufficient","n":0}
    arr=np.asarray(effects)
    return {"status":"ok","n":len(arr),"estimate":float(did_effect(v,intervention_date)["did_abs"]),"ci_low":float(np.quantile(arr,.025)),"ci_high":float(np.quantile(arr,.975)),"share_negative":float((arr<0).mean())}
