import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from app.config import CONFIG

REQUIRED=["building_id","meter","date","meter_reading","sub_primaryspaceusage","sqm"]

def load_filter(path):
    cols=REQUIRED+["site_id","airTemperature","cloudCoverage","dewTemperature","windSpeed","season"]
    # Read the header first so missing required columns produce an actionable error.
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(REQUIRED) - set(header))
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))
    df=pd.read_csv(path,usecols=lambda c:c in cols,parse_dates=["date"])
    df=df[df["meter"].eq("electricity")].copy()
    df=df[df["sub_primaryspaceusage"].fillna("").str.contains("Education|College|University|Classroom|Laboratory|School",case=False,regex=True)]
    df["meter_reading"]=pd.to_numeric(df["meter_reading"],errors="coerce")
    df["sqm"]=pd.to_numeric(df["sqm"],errors="coerce")
    return df.dropna(subset=["building_id","date","meter_reading","sqm"]).loc[lambda x:x["sqm"]>0].sort_values(["building_id","date"])

def features(df):
    d=df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["month"]=d.date.dt.month; d["dayofyear"]=d.date.dt.dayofyear; d["dow"]=d.date.dt.dayofweek; d["is_weekend"]=(d.dow>=5).astype(int)
    g=d.groupby("building_id")["meter_reading"]
    for k in [1,7,14,28]: d[f"lag{k}"]=g.shift(k)
    for w in [7,28]:
        d[f"rm{w}"]=g.transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean())
        d[f"rv{w}"]=g.transform(lambda s:s.shift(1).rolling(w,min_periods=w).std())
    # Weather fields are optional in many public building datasets. Only use
    # them when present and at least one value is available; never drop every
    # row simply because an optional weather column is absent.
    f=["sqm","month","dayofyear","dow","is_weekend","lag1","lag7","lag14","lag28","rm7","rm28","rv7","rv28"]
    for col in ["airTemperature","cloudCoverage","dewTemperature","windSpeed"]:
        if col in d.columns:
            values = pd.to_numeric(d[col], errors="coerce")
            # Keep a weather feature only when it has observed values, then
            # median-impute its missing rows so optional gaps do not discard
            # otherwise valid energy history.
            if values.notna().any():
                median_value = values.median()
                d[col] = values.fillna(median_value)
                f.append(col)
    return d,f

def deterministic_sample(train,cap=100000):
    if len(train)<=cap:return train
    take=max(1,int(cap/train["building_id"].nunique())); parts=[]
    for _,g in train.groupby("building_id",sort=False):
        idx=np.linspace(0,len(g)-1,min(take,len(g))).astype(int); parts.append(g.iloc[idx])
    s=pd.concat(parts)
    return s if len(s)<=cap else s.iloc[np.linspace(0,len(s)-1,cap).astype(int)]

def train_model(d,f,cap=100000,seed=42):
    d=d.dropna(subset=f+["meter_reading"]).copy()
    if d.empty:
        raise ValueError("No complete rows remain after feature engineering. Check required fields and history length.")
    dates=np.sort(d.date.dropna().unique())
    if len(dates)<2:
        raise ValueError("At least two distinct dates with complete features are required for chronological evaluation.")
    cutoff=pd.Timestamp(dates[int(len(dates)*.8)])
    tr=d[d.date<cutoff]; te=d[d.date>=cutoff]
    if tr.empty or te.empty:
        raise ValueError("Chronological split produced an empty train or test set. Provide more valid historical data.")
    tr_fit=deterministic_sample(tr,cap)
    if tr_fit.empty:
        raise ValueError("Training sample is empty; increase input data or check train-cap.")
    model=ExtraTreesRegressor(n_estimators=160,random_state=seed,n_jobs=-1,min_samples_leaf=2,max_features=.9)
    model.fit(tr_fit[f],tr_fit.meter_reading); pred=model.predict(te[f]); y=te.meter_reading.to_numpy()
    denominator=float(np.abs(y).sum())
    return model,tr_fit,te,pred,cutoff,{"r2":float(r2_score(y,pred)) if len(y)>=2 else float("nan"),"mae":float(mean_absolute_error(y,pred)),"wape":float(np.abs(y-pred).sum()/denominator) if denominator>0 else float("nan"),"train_rows":int(len(tr_fit)),"test_rows":int(len(te)),"cutoff":str(cutoff.date()),"baseline_benchmarks":baseline_metrics(d)}


def evaluate_unseen_buildings(d, f, test_size=0.2, seed=42, cap=100000):
    """Evaluate transfer to buildings excluded entirely from model fitting.

    Test-building rows may use their own past-only lag features, representing
    a deployment where historical meter data for a new building are available.
    This is a building-group holdout, not a cold-start test without history.
    """
    required = list(dict.fromkeys(list(f) + ["building_id", "date", "meter_reading"]))
    data = d.dropna(subset=required).copy()
    groups = data["building_id"].dropna().unique()
    if len(groups) < 2:
        raise ValueError("At least two buildings with complete feature rows are required.")
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be between 0 and 1.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, groups=data["building_id"]))
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()
    train_ids = set(train["building_id"].unique())
    test_ids = set(test["building_id"].unique())
    if train_ids & test_ids:
        raise RuntimeError("Building-group split leaked building IDs across partitions.")
    if train.empty or test.empty:
        raise ValueError("The building-group split produced an empty partition.")

    train_fit = deterministic_sample(train, cap)
    model = ExtraTreesRegressor(
        n_estimators=160,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=0.9,
    )
    model.fit(train_fit[f], train_fit["meter_reading"])
    pred = model.predict(test[f])
    y = test["meter_reading"].to_numpy()
    denom = float(np.abs(y).sum())
    metrics = {
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "wape": float(np.abs(y - pred).sum() / denom) if denom > 0 else float("nan"),
        "train_rows": int(len(train_fit)),
        "test_rows": int(len(test)),
        "train_buildings": sorted(map(str, train_ids)),
        "test_buildings": sorted(map(str, test_ids)),
        "split": "unseen_building_group_holdout",
        "test_size": float(test_size),
        "seed": int(seed),
        "history_note": "Test buildings may use their own past-only lag features; their rows are excluded from model fitting.",
    }
    return model, train_fit, test, pred, metrics

def anomaly_table(df):
    x=df[["building_id","date","meter_reading","sub_primaryspaceusage","sqm","site_id"]].copy(); g=x.groupby("building_id")["meter_reading"]
    x["baseline28"]=g.transform(lambda s:s.shift(1).rolling(28,min_periods=14).median()); x["residual"]=x.meter_reading-x.baseline28
    x["mad56"]=x.groupby("building_id")["residual"].transform(lambda s:s.shift(1).rolling(56,min_periods=14).apply(lambda v:np.median(np.abs(v-np.median(v))),raw=True))
    x["scale"]=(1.4826*x.mad56).clip(lower=1e-9); x["stable_dev"]=x.residual/x.baseline28.abs().clip(lower=1e-9); x["robust_z"]=x.residual/x.scale
    x["is_anomaly"]=((x.stable_dev>=CONFIG.relative_gap_threshold)&(x.robust_z>=2)).astype(int)
    q=x.groupby("building_id").agg(sub_primaryspaceusage=("sub_primaryspaceusage","first"),sqm=("sqm","first"),site_id=("site_id","first"),days=("date","nunique"),anomaly_days=("is_anomaly","sum"),mean_stable_dev=("stable_dev",lambda s:float(s.replace([np.inf,-np.inf],np.nan).dropna().clip(lower=0).mean()))).reset_index()
    q["anomaly_share"]=q.anomaly_days/q.days.clip(lower=1); q["priority_score"]=(.55*q.mean_stable_dev.rank(pct=True)+.45*q.anomaly_share.rank(pct=True))*100
    return q.sort_values("priority_score",ascending=False)

def carbon_scenario(df, reduction=0.20, ef=0.5306, ef_label="China 2023 location-based electricity factor"):
    """Create a conditional Scope 2 screening scenario.

    The input meter_reading is deliberately not renamed to kWh. The caller
    must independently verify meter units before interpreting the result as
    physical electricity consumption.
    """
    a = (
        df.groupby("building_id")["meter_reading"]
        .sum()
        .rename("annual_meter_reading")
        .reset_index()
    )
    a["scenario_reduction"] = a["annual_meter_reading"] * float(reduction)
    a["scenario_co2_t_if_kwh"] = a["scenario_reduction"] * float(ef) / 1000
    a["emission_factor_kgco2_per_kwh"] = float(ef)
    a["emission_factor_label"] = ef_label
    a["accounting_boundary"] = "Scope 2 · location-based screening"
    a["unit_warning"] = "Scenario only; verify meter_reading unit before interpreting as kWh."
    return a


def baseline_metrics(d, target="meter_reading"):
    """Compare simple past-only baselines on the same chronological holdout."""
    d=d.copy().sort_values(["building_id","date"])
    dates=np.sort(d.date.dropna().unique())
    if len(dates)<10: return {}
    cutoff=pd.Timestamp(dates[int(len(dates)*.8)])
    g=d.groupby("building_id")[target]
    d["naive_1"]=g.shift(1)
    d["seasonal_7"]=g.shift(7)
    d["median_28"]=g.transform(lambda s:s.shift(1).rolling(28,min_periods=7).median())
    te=d[d.date>=cutoff]
    out={}; y=te[target].to_numpy()
    for name in ["naive_1","seasonal_7","median_28"]:
        p=te[name].to_numpy(); ok=np.isfinite(p)&np.isfinite(y)
        if ok.sum():
            out[name]={"mae":float(mean_absolute_error(y[ok],p[ok])),
                       "wape":float(np.abs(y[ok]-p[ok]).sum()/max(np.abs(y[ok]).sum(),1e-9))}
    return out
