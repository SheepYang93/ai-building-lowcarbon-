import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.pipeline import load_filter,features,train_model,anomaly_table,carbon_scenario
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="outputs"); p.add_argument("--train-cap",type=int,default=100000); args=p.parse_args()
out=Path(args.output); (out/"figures").mkdir(parents=True,exist_ok=True)
df=load_filter(args.input); d,f=features(df); model,tr,te,pred,cut,metrics=train_model(d,f,args.train_cap)
metrics.update({"rows_after_filter":len(df),"buildings":df.building_id.nunique(),"date_start":str(df.date.min().date()),"date_end":str(df.date.max().date()),"model":"ExtraTreesRegressor","train_cap":args.train_cap})
(out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)); pd.DataFrame({"feature":f,"importance":model.feature_importances_}).sort_values("importance",ascending=False).to_csv(out/"feature_importance.csv",index=False)
an=anomaly_table(df); an.to_csv(out/"anomaly_buildings.csv",index=False); carbon_scenario(df).to_csv(out/"carbon_scenarios.csv",index=False)
sample=te.sample(min(5000,len(te)),random_state=42); pos=te.index.get_indexer(sample.index); yy=te.meter_reading.to_numpy()[pos]; pp=pred[pos]
plt.figure(figsize=(8,5)); plt.scatter(yy,pp,s=8,alpha=.3); lo=min(yy.min(),pp.min()); hi=max(yy.max(),pp.max()); plt.plot([lo,hi],[lo,hi]); plt.xlabel("Actual meter reading"); plt.ylabel("Predicted meter reading"); plt.title("Strict chronological test split"); plt.tight_layout(); plt.savefig(out/"figures/prediction_vs_actual.png",dpi=160); plt.close()
