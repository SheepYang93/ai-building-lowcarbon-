import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.pipeline import load_filter, features, train_model, evaluate_unseen_buildings, anomaly_table, carbon_scenario

p=argparse.ArgumentParser()
p.add_argument("--input",required=True)
p.add_argument("--output",default="outputs")
p.add_argument("--train-cap",type=int,default=100000)
p.add_argument("--building-test-size",type=float,default=0.2,help="Fraction of buildings held out for generalization evaluation")
p.add_argument("--building-seed",type=int,default=42)
args=p.parse_args()

out=Path(args.output)
(out/"figures").mkdir(parents=True,exist_ok=True)

if not Path(args.input).is_file():
    p.error(f"Input CSV not found: {args.input}. Place the BDG2-derived dataset there or pass --input PATH.")

df=load_filter(args.input)
if df.empty:
    p.error("No rows remain after filtering. Check meter type, building-use values, and required data columns.")
d,f=features(df)
model,tr,te,pred,cut,metrics=train_model(d,f,args.train_cap)

# Separate group holdout: test buildings are never used to fit this model.
# Their own past-only lag features remain available, so this is not a cold-start test.
try:
    _, building_tr, building_te, building_pred, building_metrics = evaluate_unseen_buildings(
        d, f, test_size=args.building_test_size, seed=args.building_seed, cap=args.train_cap
    )
    metrics["unseen_building_evaluation"] = building_metrics
    pd.DataFrame({
        "building_id": building_te["building_id"].astype(str).to_numpy(),
        "date": building_te["date"].to_numpy(),
        "actual": building_te["meter_reading"].to_numpy(),
        "predicted": building_pred,
    }).to_csv(out/"unseen_building_predictions.csv", index=False)
except ValueError as exc:
    metrics["unseen_building_evaluation"] = {"status": "not_run", "reason": str(exc)}

metrics.update({
    "rows_after_filter":len(df),
    "buildings":df.building_id.nunique(),
    "date_start":str(df.date.min().date()),
    "date_end":str(df.date.max().date()),
    "model":"ExtraTreesRegressor",
    "train_cap":args.train_cap
})
(out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
pd.DataFrame({"feature":f,"importance":model.feature_importances_}).sort_values("importance",ascending=False).to_csv(out/"feature_importance.csv",index=False)

an=anomaly_table(df)
an.to_csv(out/"anomaly_buildings.csv",index=False)
carbon_scenario(df, reduction=0.20, ef=0.5306, ef_label="China 2023 location-based electricity factor").to_csv(out/"carbon_scenarios.csv",index=False)

if len(te):
    sample=te.sample(min(5000,len(te)),random_state=42)
    pos=te.index.get_indexer(sample.index)
    yy=te.meter_reading.to_numpy()[pos]
    pp=pred[pos]

    plt.figure(figsize=(8,5))
    plt.scatter(yy,pp,s=8,alpha=.3)
    lo=min(yy.min(),pp.min())
    hi=max(yy.max(),pp.max())
    plt.plot([lo,hi],[lo,hi])
    plt.xlabel("Actual meter reading")
    plt.ylabel("Predicted meter reading")
    plt.title("Strict chronological test split")
    plt.tight_layout()
    plt.savefig(out/"figures/prediction_vs_actual.png",dpi=160)
    plt.close()
else:
    print("Warning: chronological test set is empty; prediction plot was skipped.")
