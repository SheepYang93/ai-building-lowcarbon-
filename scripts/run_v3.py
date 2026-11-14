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

if args.train_cap <= 0:
    p.error("--train-cap must be a positive integer.")
if not 0 < args.building_test_size < 1:
    p.error("--building-test-size must be strictly between 0 and 1.")

out=Path(args.output)
(out/"figures").mkdir(parents=True,exist_ok=True)

if not Path(args.input).is_file():
    p.error(f"Input CSV not found: {args.input}. Place the BDG2-derived dataset there or pass --input PATH.")

df=load_filter(args.input)
if df.empty:
    p.error("No rows remain after filtering. Check meter type, building-use values, and required data columns.")
d,f=features(df)
model,tr,te,pred,cut,metrics=train_model(d,f,args.train_cap)
if te.empty or len(pred) != len(te):
    p.error(
        f"Chronological evaluation is incomplete: test_rows={len(te)}, predictions={len(pred)}. "
        "No evaluation report was exported."
    )

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
    }).to_csv(out/"unseen_building_predictions.csv", index=False, na_rep="N/A")
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
actual_abs = chronological_predictions["meter_reading"].abs()
chronological_predictions["absolute_percentage_error"] = np.where(
    actual_abs > 0, chronological_predictions["absolute_error"] / actual_abs, np.nan
)
chronological_predictions.to_csv(out/"chronological_test_predictions.csv", index=False, na_rep="N/A")
from src.reporting import building_error_summary
error_by_building = building_error_summary(
    chronological_predictions.rename(columns={"meter_reading": "actual"})[["building_id", "actual", "predicted"]]
)
error_by_building.to_csv(out/"error_analysis_by_building.csv", index=False, na_rep="N/A")
error_by_building.head(10).to_csv(out/"top10_high_error_buildings.csv", index=False, na_rep="N/A")

(out/"metrics.json").write_text(json.dumps(json_safe(metrics),ensure_ascii=False,indent=2,allow_nan=False))

# Machine-readable metrics above are paired with a recruiter-friendly Markdown summary.
report_path = out/"evaluation_report.md"
bm = metrics.get("baseline_benchmarks", {})
unseen = metrics.get("unseen_building_evaluation", {})
def fmt(x):
    try:
        value = float(x)
        return f"{value:.4f}" if np.isfinite(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"
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
    "| Baseline | MAE | RMSE | WAPE | R² | n | Coverage |", "|---|---:|---:|---:|---:|---:|---:|",
]
for name, item in bm.items():
    lines.append(f"| {name} | {fmt(item.get('mae'))} | {fmt(item.get('rmse'))} | {fmt(item.get('wape'))} | {fmt(item.get('r2'))} | {item.get('n','N/A')} | {fmt(item.get('coverage'))} |")
common = metrics.get("common_baseline_comparison", {})
cm = common.get("model", {})
cb = common.get("baselines", {})
lines += ["", "## Fair comparison on identical valid rows",
    f"- Shared rows: {cm.get('n','N/A')} / {metrics.get('test_rows','N/A')} ({fmt(cm.get('coverage_of_chronological_test'))} coverage)",
    "| Model / baseline | MAE | RMSE | WAPE | R² | n |", "|---|---:|---:|---:|---:|---:|",
    f"| ExtraTrees | {fmt(cm.get('mae'))} | {fmt(cm.get('rmse'))} | {fmt(cm.get('wape'))} | {fmt(cm.get('r2'))} | {cm.get('n','N/A')} |"]
for name, item in cb.items():
    lines.append(f"| {name} | {fmt(item.get('mae'))} | {fmt(item.get('rmse'))} | {fmt(item.get('wape'))} | {fmt(item.get('r2'))} | {item.get('n','N/A')} |")
lines += ["", "## Building-group holdout",
    f"- Status: {unseen.get('status', 'ok' if 'r2' in unseen else 'not_run')}",
    f"- R² / MAE / RMSE / WAPE: {fmt(unseen.get('r2'))} / {fmt(unseen.get('mae'))} / {fmt(unseen.get('rmse'))} / {fmt(unseen.get('wape'))}",
    f"- Test rows / buildings: {unseen.get('test_rows','N/A')} / {len(unseen.get('test_buildings', []))}",
    f"- Note: {unseen.get('history_note', unseen.get('reason',''))}", "",
    "## Building-level error diagnostics",
    f"- Buildings with valid chronological test predictions: {len(error_by_building)}",
    "- Residual convention: actual minus predicted; positive means underprediction, negative means overprediction.",
    "- Per-building MAE is sorted descending in error_analysis_by_building.csv; top 10 are in top10_high_error_buildings.csv.",
]
if not error_by_building.empty:
    worst = error_by_building.iloc[0]
    lines.append(
        f"- Highest-MAE building in this split: {worst['building_id']} "
        f"(MAE {fmt(worst['mae'])}; bias {worst['bias_direction']}; "
        f"test rows {int(worst['test_rows'])})."
    )
lines += ["", "## Exported artifacts",
    "- metrics.json: machine-readable metrics (undefined values are null).",
    "- chronological_test_predictions.csv: row-level actual, predicted and residual values.",
    "- unseen_building_predictions.csv: separate building-group holdout predictions, when available.",
    "- error_analysis_by_building.csv / top10_high_error_buildings.csv: building-level diagnostics.",
    "- feature_importance.csv, anomaly_buildings.csv and carbon_scenarios.csv: supporting analysis outputs.",
    "", "## Limits",
    "- Building holdout is not cold-start: held-out buildings can use their own past meter history.",
    "- Meter-reading units must be verified before interpreting predictions or scenario CO₂ as physical kWh / emissions.",
    "- Carbon scenarios assume a stated reduction and emission factor; they are not measured savings.",
    "- Metrics depend on filtering, time window, building mix and meter-unit lineage.",
    "- Predictive accuracy does not establish energy savings or causal intervention effects.", ""]
report_path.write_text("\n".join(lines), encoding="utf-8")

# Export a portable Word version through a separately testable reporting module.
from src.reporting import markdown_to_docx
markdown_to_docx(report_path, out/"evaluation_report.docx")

print(f"Evaluation report written to: {report_path}")
pd.DataFrame({"feature":f,"importance":model.feature_importances_}).sort_values("importance",ascending=False).to_csv(out/"feature_importance.csv",index=False,na_rep="N/A")

an=anomaly_table(df)
an.to_csv(out/"anomaly_buildings.csv",index=False,na_rep="N/A")
carbon_scenario(df, reduction=0.20, ef=0.5306, ef_label="China 2023 location-based electricity factor").to_csv(out/"carbon_scenarios.csv",index=False,na_rep="N/A")

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

# Record provenance for reproducibility without exposing local absolute paths.
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
    ).strip()
except (subprocess.SubprocessError, FileNotFoundError):
    git_commit = None

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

input_path = Path(args.input)
input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
manifest = {
    "manifest_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit,
    "input": {
        "filename": input_path.name,
        "sha256": input_sha256,
        "rows_after_filter": int(len(df)),
        "buildings_after_filter": int(df["building_id"].nunique()),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
    },
    "parameters": {
        "train_cap": args.train_cap,
        "building_test_size": args.building_test_size,
        "building_seed": args.building_seed,
        "model": "ExtraTreesRegressor",
    },
    "runtime": {
        "python": platform.python_version(),
        "pandas": package_version("pandas"),
        "numpy": package_version("numpy"),
        "scikit_learn": package_version("scikit-learn"),
    },
    "artifacts": [
        "metrics.json", "evaluation_report.md", "evaluation_report.docx",
        "chronological_test_predictions.csv", "unseen_building_predictions.csv",
        "error_analysis_by_building.csv", "top10_high_error_buildings.csv",
        "feature_importance.csv", "anomaly_buildings.csv",
        "carbon_scenarios.csv", "figures/prediction_vs_actual.png",
    ],
}
(out / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
)
print(f"Run manifest written to: {out / 'run_manifest.json'}")
