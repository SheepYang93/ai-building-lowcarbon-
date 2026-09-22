import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

REQUIRED=["building_id","meter","date","meter_reading","sub_primaryspaceusage","sqm"]

def load_filter(path):
    cols=REQUIRED+["site_id","airTemperature","cloudCoverage","dewTemperature","windSpeed","season"]
    df=pd.read_csv(path,usecols=lambda c:c in cols,parse_dates=["date"])
    df=df[df["meter"].eq("electricity")].copy()
    df=df[df["sub_primaryspaceusage"].fillna("").str.contains("Education|College|University|Classroom|Laboratory|School",case=False,regex=True)]
    df["meter_reading"]=pd.to_numeric(df["meter_reading"],errors="coerce")
    df["sqm"]=pd.to_numeric(df["sqm"],errors="coerce")
    return df.dropna(subset=["building_id","date","meter_reading","sqm"]).loc[lambda x:x["sqm"]>0].sort_values(["building_id","date"])

def features(df):
    d=df.copy(); d["month"]=d.date.dt.month; d["dayofyear"]=d.date.dt.dayofyear; d["dow"]=d.date.dt.dayofweek; d["is_weekend"]=(d.dow>=5).astype(int)
    g=d.groupby("building_id")["meter_reading"]
    for k in [1,7,14,28]: d[f"lag{k}"]=g.shift(k)
    for w in [7,28]:
        d[f"rm{w}"]=g.transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean())
        d[f"rv{w}"]=g.transform(lambda s:s.shift(1).rolling(w,min_periods=w).std())
    f=["sqm","month","dayofyear","dow","is_weekend","airTemperature","cloudCoverage","dewTemperature","windSpeed","lag1","lag7","lag14","lag28","rm7","rm28","rv7","rv28"]
    return d,f

def deterministic_sample(train,cap=100000):
    if len(train)<=cap:return train
    take=max(1,int(cap/train["building_id"].nunique())); parts=[]
    for _,g in train.groupby("building_id",sort=False):
        idx=np.linspace(0,len(g)-1,min(take,len(g))).astype(int); parts.append(g.iloc[idx])
    s=pd.concat(parts)
    return s if len(s)<=cap else s.iloc[np.linspace(0,len(s)-1,cap).astype(int)]

def train_model(d,f,cap=100000,seed=42):
    d=d.dropna(subset=f+["meter_reading"]).copy(); dates=np.sort(d.date.unique()); cutoff=pd.Timestamp(dates[int(len(dates)*.8)])
    tr=d[d.date<cutoff]; te=d[d.date>=cutoff]; tr_fit=deterministic_sample(tr,cap)
    model=ExtraTreesRegressor(n_estimators=160,random_state=seed,n_jobs=-1,min_samples_leaf=2,max_features=.9)
    model.fit(tr_fit[f],tr_fit.meter_reading); pred=model.predict(te[f]); y=te.meter_reading.to_numpy()
    return model,tr_fit,te,pred,cutoff,{"r2":float(r2_score(y,pred)),"mae":float(mean_absolute_error(y,pred)),"wape":float(np.abs(y-pred).sum()/np.abs(y).sum()),"train_rows":int(len(tr_fit)),"test_rows":int(len(te)),"cutoff":str(cutoff.date())}

def anomaly_table(df):
    x=df[["building_id","date","meter_reading","sub_primaryspaceusage","sqm","site_id"]].copy(); g=x.groupby("building_id")["meter_reading"]
    x["baseline28"]=g.transform(lambda s:s.shift(1).rolling(28,min_periods=14).median()); x["residual"]=x.meter_reading-x.baseline28
    x["mad56"]=x.groupby("building_id")["residual"].transform(lambda s:s.shift(1).rolling(56,min_periods=14).apply(lambda v:np.median(np.abs(v-np.median(v))),raw=True))
    x["scale"]=(1.4826*x.mad56).clip(lower=1e-9); x["stable_dev"]=x.residual/x.baseline28.abs().clip(lower=1e-9); x["robust_z"]=x.residual/x.scale
    x["is_anomaly"]=((x.stable_dev>=.20)&(x.robust_z>=2)).astype(int)
    q=x.groupby("building_id").agg(sub_primaryspaceusage=("sub_primaryspaceusage","first"),sqm=("sqm","first"),site_id=("site_id","first"),days=("date","nunique"),anomaly_days=("is_anomaly","sum"),mean_stable_dev=("stable_dev",lambda s:float(s.replace([np.inf,-np.inf],np.nan).dropna().clip(lower=0).mean()))).reset_index()
    q["anomaly_share"]=q.anomaly_days/q.days.clip(lower=1); q["priority_score"]=(.55*q.mean_stable_dev.rank(pct=True)+.45*q.anomaly_share.rank(pct=True))*100
    return q.sort_values("priority_score",ascending=False)

def carbon_scenario(df,reduction=.20,ef=.5777):
    a=df.groupby("building_id").meter_reading.sum().rename("annual_reading").reset_index(); a["scenario_reduction"]=a.annual_reading*reduction; a["scenario_co2e_t_if_kwh"]=a.scenario_reduction*ef/1000; a["unit_warning"]="Scenario only; verify meter_reading unit before interpreting as kWh."; return a
