import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
    df=pd.read_csv(path,usecols=lambda c:c in cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df=df[df["meter"].eq("electricity")].copy()
    df=df[df["sub_primaryspaceusage"].fillna("").str.contains("Education|College|University|Classroom|Laboratory|School",case=False,regex=True)]
    df["meter_reading"]=pd.to_numeric(df["meter_reading"],errors="coerce")
    df["sqm"]=pd.to_numeric(df["sqm"],errors="coerce")
    df = df.dropna(subset=["building_id","date","meter_reading","sqm"]).loc[lambda x:x["sqm"]>0]
    duplicates = df.duplicated(["building_id", "date"], keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, ["building_id", "date"]].head(5).to_dict("records")
        raise ValueError(
            f"Input contains {int(duplicates.sum())} duplicate building/date readings; "
            f"daily modeling requires one row per building per date. Examples: {sample}"
        )
    return df.sort_values(["building_id","date"])


def features(df):
    d=df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d=d.dropna(subset=["building_id","date","meter_reading","sqm"]).copy()
    d=d.sort_values(["building_id","date"],kind="mergesort").reset_index(drop=True)
    d["month"]=d.date.dt.month; d["dayofyear"]=d.date.dt.dayofyear; d["dow"]=d.date.dt.dayofweek; d["is_weekend"]=(d.dow>=5).astype(int)

    # Build one continuous daily calendar per building. Missing dates stay NaN,
    # so lags and rolling features represent calendar days rather than row counts.
    for _, indices in d.groupby("building_id", sort=False).groups.items():
        group=d.loc[indices]
        observed=group.set_index("date")["meter_reading"].sort_index()
        calendar=pd.date_range(observed.index.min(), observed.index.max(), freq="D")
        daily=observed.reindex(calendar)
        dates=group["date"]
        for k in [1,7,14,28]:
            d.loc[indices, f"lag{k}"]=dates.map(daily.shift(k)).to_numpy()
        for w in [7,28]:
            rolling=daily.shift(1).rolling(w, min_periods=w)
            d.loc[indices, f"rm{w}"]=dates.map(rolling.mean()).to_numpy()
            d.loc[indices, f"rv{w}"]=dates.map(rolling.std()).to_numpy()

    # Weather fields are optional. Include only complete numeric fields to avoid
    # full-timeline median imputation leaking future information into time splits.
    f=["sqm","month","dayofyear","dow","is_weekend","lag1","lag7","lag14","lag28","rm7","rm28","rv7","rv28"]
    for col in ["airTemperature","cloudCoverage","dewTemperature","windSpeed"]:
        if col in d.columns:
            values = pd.to_numeric(d[col], errors="coerce")
            if values.notna().all():
                d[col] = values
                f.append(col)
    return d,f

def deterministic_sample(train,cap=100000):
    if len(train)<=cap:return train
    take=max(1,int(cap/train["building_id"].nunique())); parts=[]
    for _,g in train.groupby("building_id",sort=False):
        idx=np.linspace(0,len(g)-1,min(take,len(g))).astype(int); parts.append(g.iloc[idx])
    s=pd.concat(parts)
    return s if len(s)<=cap else s.iloc[np.linspace(0,len(s)-1,cap).astype(int)]

def _regression_metrics(y, pred):
    """Return stable regression metrics with explicit undefined-value behavior."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.ndim != 1 or pred.ndim != 1 or len(y) != len(pred):
        raise ValueError("y and predictions must be one-dimensional arrays of equal length.")
    if len(y) == 0:
        raise ValueError("Cannot evaluate metrics on an empty cohort.")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must contain only finite values.")
    error = y - pred
    denominator = float(np.abs(y).sum())
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "wape": float(np.abs(error).sum() / denominator) if denominator > 0 else float("nan"),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "n": int(len(y)),
        "wape_status": "ok" if denominator > 0 else "undefined_zero_actual_sum",
        "r2_status": "ok" if len(y) >= 2 else "undefined_single_sample",
    }


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
    model_metrics=_regression_metrics(y,pred)
    baseline_frame=baseline_predictions(d,te)
    baseline_names=["naive_1d","seasonal_7d","median_28"]
    baseline_matrix=baseline_frame[baseline_names].to_numpy(dtype=float)
    common=np.isfinite(y)&np.isfinite(pred)&np.isfinite(baseline_matrix).all(axis=1)
    common_y=y[common]; common_pred=pred[common]
    common_coverage=float(common.sum()/len(te)) if len(te) else 0.0
    if len(common_y):
        common_metrics=_regression_metrics(common_y,common_pred)
        common_baselines={name:_regression_metrics(common_y,baseline_matrix[common,j])
                          for j,name in enumerate(baseline_names)}
    else:
        common_metrics={"mae":np.nan,"rmse":np.nan,"wape":np.nan,"r2":np.nan,"n":0,
                        "status":"no_shared_valid_rows","wape_status":"not_evaluated",
                        "r2_status":"not_evaluated"}
        common_baselines={name:{**common_metrics} for name in baseline_names}
    common_metrics["coverage_of_chronological_test"]=common_coverage
    return model,tr_fit,te,pred,cutoff,{
        **model_metrics,"train_rows":int(len(tr_fit)),"test_rows":int(len(te)),"cutoff":str(cutoff.date()),
        "baseline_benchmarks":baseline_metrics(d,evaluation_frame=te),"baseline_comparison_rows":int(len(te)),
        "common_baseline_comparison":{"evaluation_cohort":"intersection of rows with valid ExtraTrees and all three baseline predictions","model":common_metrics,"baselines":common_baselines},
        "meter_unit_status":"UNVERIFIED: meter_reading unit lineage must be confirmed before kWh or CO2 interpretation"
    }

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
    metrics = _regression_metrics(test["meter_reading"].to_numpy(), pred)
    metrics.update({
        "train_rows": int(len(train_fit)),
        "test_rows": int(len(test)),
        "train_buildings": sorted(map(str, train_ids)),
        "test_buildings": sorted(map(str, test_ids)),
        "split": "unseen_building_group_holdout",
        "test_size": float(test_size),
        "seed": int(seed),
        "history_note": "Test buildings may use their own past-only lag features; their rows are excluded from model fitting.",
    })
    return model, train_fit, test, pred, metrics

def anomaly_table(df):
    # site_id is optional in the accepted input schema; preserve diagnosis
    # when a dataset has no site-level identifier.
    x = df.copy()
    if "site_id" not in x.columns:
        x["site_id"] = pd.NA
    required = ["building_id", "date", "meter_reading", "sub_primaryspaceusage", "sqm", "site_id"]
    missing = [column for column in required if column not in x.columns]
    if missing:
        raise ValueError("Anomaly diagnosis is missing required columns: " + ", ".join(missing))
    x = x[required].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["meter_reading"] = pd.to_numeric(x["meter_reading"], errors="coerce")
    x = x.dropna(subset=["building_id", "date", "meter_reading"])
    x = x.sort_values(["building_id", "date"]).reset_index(drop=True)

    # Rolling reference windows are calendar-exact; gaps do not compress time.
    x["baseline28"] = np.nan
    x["mad56"] = np.nan
    for _, indices in x.groupby("building_id", sort=False).groups.items():
        group=x.loc[indices]
        observed=group.set_index("date")["meter_reading"].sort_index()
        calendar=pd.date_range(observed.index.min(), observed.index.max(), freq="D")
        daily=observed.reindex(calendar)
        baseline=daily.shift(1).rolling(28, min_periods=14).median()
        residual=daily-baseline
        mad=residual.shift(1).rolling(
            56, min_periods=14
        ).apply(lambda values: np.median(np.abs(values-np.median(values))), raw=True)
        dates=group["date"]
        x.loc[indices,"baseline28"]=dates.map(baseline).to_numpy()
        x.loc[indices,"mad56"]=dates.map(mad).to_numpy()

    x["residual"]=x.meter_reading-x.baseline28
    x["scale"]=(1.4826*x.mad56).clip(lower=1e-9)
    x["stable_dev"]=x.residual/x.baseline28.abs().clip(lower=1e-9)
    x["robust_z"]=x.residual/x.scale
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
        .rename("period_meter_reading")
        .reset_index()
    )
    a["scenario_reduction"] = a["period_meter_reading"] * float(reduction)
    a["scenario_co2_t_if_kwh"] = a["scenario_reduction"] * float(ef) / 1000
    a["emission_factor_kgco2_per_kwh"] = float(ef)
    a["emission_factor_label"] = ef_label
    a["accounting_boundary"] = "Scope 2 · location-based screening"
    a["unit_warning"] = "Scenario only; verify meter_reading unit before interpreting as kWh."
    return a


def baseline_predictions(d, evaluation_frame):
    """Return row-aligned, past-only baseline predictions for an explicit evaluation cohort."""
    history = d.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["meter_reading"] = pd.to_numeric(history["meter_reading"], errors="coerce")
    history = history.dropna(subset=["building_id", "date", "meter_reading"]).sort_values(
        ["building_id", "date"], kind="mergesort"
    )
    if history.duplicated(["building_id", "date"]).any():
        raise ValueError("Baseline history must contain at most one reading per building/date.")
    evaluation = evaluation_frame[["building_id", "date", "meter_reading"]].copy()
    evaluation["date"] = pd.to_datetime(evaluation["date"], errors="coerce").dt.normalize()
    if not evaluation.index.is_unique:
        raise ValueError("Evaluation frame must have unique row indices.")
    lookup = history.set_index(["building_id", "date"])["meter_reading"]
    evaluation["naive_1d"] = [
        lookup.get((bid, date - pd.Timedelta(days=1)), np.nan)
        for bid, date in zip(evaluation["building_id"], evaluation["date"])
    ]
    evaluation["seasonal_7d"] = [
        lookup.get((bid, date - pd.Timedelta(days=7)), np.nan)
        for bid, date in zip(evaluation["building_id"], evaluation["date"])
    ]
    medians = []
    grouped = {bid: group for bid, group in history.groupby("building_id", sort=False)}
    for bid, date in zip(evaluation["building_id"], evaluation["date"]):
        group = grouped.get(bid)
        if group is None or pd.isna(date):
            medians.append(np.nan)
            continue
        past = group.loc[(group["date"] >= date-pd.Timedelta(days=28)) & (group["date"] < date), "meter_reading"]
        medians.append(float(past.median()) if len(past) >= 7 else np.nan)
    evaluation["median_28"] = medians
    return evaluation


def baseline_metrics(d, target="meter_reading", evaluation_frame=None):
    """Score past-only baselines; each metric reports its own valid-row coverage."""
    history = d.copy()
    history[target] = pd.to_numeric(history[target], errors="coerce")
    if target != "meter_reading":
        history = history.rename(columns={target: "meter_reading"})
    if evaluation_frame is None:
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
        dates = np.sort(history["date"].dropna().unique())
        if len(dates) < 10:
            return {}
        cutoff = pd.Timestamp(dates[int(len(dates)*0.8)])
        evaluation_frame = history.loc[history["date"] >= cutoff].copy()
    else:
        evaluation_frame = evaluation_frame.copy()
    preds = baseline_predictions(history, evaluation_frame)
    y = pd.to_numeric(preds["meter_reading"], errors="coerce").to_numpy(dtype=float)
    out = {}
    for name in ["naive_1d", "seasonal_7d", "median_28"]:
        p = preds[name].to_numpy(dtype=float)
        ok = np.isfinite(p) & np.isfinite(y)
        n = int(ok.sum())
        if n:
            score = _regression_metrics(y[ok], p[ok])
            score["coverage"] = float(n / len(preds)) if len(preds) else 0.0
            score["valid_rows"] = n
            score["total_rows"] = int(len(preds))
            out[name] = score
        else:
            out[name] = {"n": 0, "valid_rows": 0, "total_rows": int(len(preds)),
                         "coverage": 0.0, "status": "no_valid_predictions",
                         "mae": np.nan, "rmse": np.nan, "wape": np.nan, "r2": np.nan}
    return out


def json_safe(value):
    """Recursively convert non-finite metrics to JSON null and NumPy scalars to native types."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value
