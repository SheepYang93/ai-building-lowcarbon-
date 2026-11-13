import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pipeline import load_filter, features, train_model, evaluate_unseen_buildings, anomaly_table, carbon_scenario, json_safe

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
# Export row-level chronological residuals and per-building error summaries.
chronological_predictions = te[["building_id", "date", "meter_reading"]].copy()
chronological_predictions["predicted"] = pred
chronological_predictions["residual"] = chronological_predictions["meter_reading"] - chronological_predictions["predicted"]
chronological_predictions["absolute_error"] = chronological_predictions["residual"].abs()
chronological_predictions["absolute_percentage_error"] = (
    chronological_predictions["absolute_error"] / chronological_predictions["meter_reading"].abs().clip(lower=1e-9)
)
chronological_predictions.to_csv(out/"chronological_test_predictions.csv", index=False)
error_by_building = chronological_predictions.groupby("building_id").agg(
    test_rows=("residual", "size"),
    mae=("absolute_error", "mean"),
    rmse=("residual", lambda s: float((s.pow(2).mean()) ** 0.5)),
    mean_error=("residual", "mean"),
    wape=("absolute_error", lambda s: float(s.sum() / max(chronological_predictions.loc[s.index, "meter_reading"].abs().sum(), 1e-9))),
    median_absolute_percentage_error=("absolute_percentage_error", "median"),
).reset_index().sort_values("mae", ascending=False)
error_by_building["bias_direction"] = np.where(error_by_building["mean_error"] > 0, "underprediction", np.where(error_by_building["mean_error"] < 0, "overprediction", "balanced"))
error_by_building.to_csv(out/"error_analysis_by_building.csv", index=False)
error_by_building.head(10).to_csv(out/"top10_high_error_buildings.csv", index=False)

(out/"metrics.json").write_text(json.dumps(json_safe(metrics),ensure_ascii=False,indent=2,allow_nan=False))

# Machine-readable metrics above are paired with a recruiter-friendly Markdown summary.
report_path = out/"evaluation_report.md"
bm = metrics.get("baseline_benchmarks", {})
unseen = metrics.get("unseen_building_evaluation", {})
fmt = lambda x: "N/A" if x is None or not np.isfinite(float(x)) else f"{float(x):.4f}"
lines = [
    "# Building Energy Model Evaluation", "",
    "> Auto-generated; results apply only to this dataset and split.", "",
    "## Dataset",
    f"- Rows: {metrics['rows_after_filter']:,}",
    f"- Buildings: {metrics['buildings']:,}",
    f"- Period: {metrics['date_start']} to {metrics['date_end']}",
    f"- Model: {metrics['model']}",
    f"- Chronological cutoff: {metrics.get('cutoff', 'N/A')}", "",
    "## Chronological holdout",
    "| Metric | Value |", "|---|---:|",
    f"| R² | {fmt(metrics.get('r2'))} |",
    f"| MAE | {fmt(metrics.get('mae'))} |",
    f"| RMSE | {fmt(metrics.get('rmse'))} |",
    f"| WAPE | {fmt(metrics.get('wape'))} |",
    f"| Test rows | {metrics.get('test_rows', 'N/A')} |", "",
    "## Baselines on the same test rows (baseline n can be smaller if history is unavailable)",
    "| Baseline | MAE | RMSE | WAPE | n |", "|---|---:|---:|---:|---:|",
]
for name, item in bm.items():
    lines.append(f"| {name} | {fmt(item.get('mae'))} | {fmt(item.get('rmse'))} | {fmt(item.get('wape'))} | {item.get('n','N/A')} |")
common = metrics.get("common_baseline_comparison", {})
cm = common.get("model", {})
cb = common.get("baselines", {})
lines += ["", "## Fair comparison on identical valid rows",
    f"- Shared rows: {cm.get('n','N/A')} / {metrics.get('test_rows','N/A')} ({fmt(cm.get('coverage_of_chronological_test'))} coverage)",
    "| Model / baseline | MAE | RMSE | WAPE | n |", "|---|---:|---:|---:|---:|",
    f"| ExtraTrees | {fmt(cm.get('mae'))} | {fmt(cm.get('rmse'))} | {fmt(cm.get('wape'))} | {cm.get('n','N/A')} |"]
for name, item in cb.items():
    lines.append(f"| {name} | {fmt(item.get('mae'))} | {fmt(item.get('rmse'))} | {fmt(item.get('wape'))} | {item.get('n','N/A')} |")
lines += ["", "## Building-group holdout",
    f"- Status: {unseen.get('status', 'ok' if 'r2' in unseen else 'not_run')}",
    f"- R² / MAE / WAPE: {fmt(unseen.get('r2'))} / {fmt(unseen.get('mae'))} / {fmt(unseen.get('wape'))}",
    f"- Test rows / buildings: {unseen.get('test_rows','N/A')} / {len(unseen.get('test_buildings', []))}",
    f"- Note: {unseen.get('history_note', unseen.get('reason',''))}", "",
    "## Limits",
    "- Building holdout is not cold-start: held-out buildings can use their own past meter history.",
    "- Metrics depend on filtering, time window, building mix and meter-unit lineage.",
    "- Predictive accuracy does not establish energy savings or causal intervention effects.", ""]
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Evaluation report written to: {report_path}")
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
