import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Building Low-carbon Decision System", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ADV = OUT / "advanced"
EXAMPLES = ROOT / "examples"

st.title("AI Building Energy Anomaly & Low-carbon Decision System")
st.caption(
    "Portfolio prototype · prediction → diagnosis → counterfactual → "
    "intervention → carbon scenario"
)

metrics_path = OUT / "metrics.json"
demo_metrics_path = EXAMPLES / "metrics_2017_public_data.json"
metrics = json.loads(
    (metrics_path if metrics_path.exists() else demo_metrics_path).read_text(
        encoding="utf-8"
    )
)

anomaly_path = OUT / "anomaly_buildings.csv"
an = pd.read_csv(anomaly_path) if anomaly_path.exists() else pd.DataFrame()

tabs = st.tabs(["Overview", "Building Diagnosis", "Counterfactual", "Intervention", "Carbon"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buildings", metrics.get("buildings", "—"))
    c2.metric("Test R²", f'{metrics["r2"]:.3f}' if "r2" in metrics else "—")
    c3.metric("WAPE", f'{metrics["wape"] * 100:.1f}%' if "wape" in metrics else "—")

    demo_cf_path = EXAMPLES / "counterfactual_candidates_demo.csv"
    demo_cf = pd.read_csv(demo_cf_path) if demo_cf_path.exists() else pd.DataFrame()
    candidate_count = (
        int((an.anomaly_days > 0).sum())
        if len(an) and "anomaly_days" in an.columns
        else len(demo_cf)
    )
    c4.metric("Screening candidates", candidate_count)

    if len(an):
        st.success("Live pipeline artifacts detected.")
    else:
        st.info(
            "Demo mode: curated public-data artifacts are displayed. "
            "Run the pipeline with your own dataset to populate live outputs."
        )

    st.markdown(
        "**Raw data → Feature engineering → ML prediction → Robust anomaly detection → "
        "Energy fingerprint → Counterfactual → Intervention matching → Carbon scenario → "
        "Field validation**"
    )
    st.info(
        "Public-data results are screening evidence. Scenario CO₂e is not measured "
        "emissions, and predicted savings require field validation."
    )

with tabs[1]:
    if len(an):
        q = an.copy()
        q["label"] = (
            q.building_id.astype(str)
            + " · "
            + q.sub_primaryspaceusage.astype(str)
        )
        label = st.selectbox("Building", q.label.head(200).tolist())
        r = q[q.label.eq(label)].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Priority", f"{r.priority_score:.1f}")
        b.metric("Anomaly days", int(r.anomaly_days))
        c.metric("Anomaly share", f"{r.anomaly_share * 100:.1f}%")
        st.write(
            {
                "building_id": r.building_id,
                "building_type": r.sub_primaryspaceusage,
                "area_m2": round(float(r.sqm), 1),
                "site_id": r.site_id,
            }
        )
    else:
        st.info(
            "Live building-diagnosis output is not bundled in the public demo. "
            "Use Counterfactual and Intervention for the curated examples."
        )

with tabs[2]:
    cf = ADV / "ai_exp45_robust_candidates.csv"
    source = "pipeline output"
    if not cf.exists():
        cf = EXAMPLES / "counterfactual_candidates_demo.csv"
        source = "curated public-data example"

    if cf.exists():
        d = pd.read_csv(cf)
        st.write(
            f"Counterfactual candidate records: **{len(d)}** · source: **{source}**"
        )
        cols = [
            c
            for c in [
                "building_id",
                "building_type",
                "protected_calibrated_pct",
                "robust_low_pct",
                "robust_high_pct",
                "robust_confidence",
                "robust_class",
            ]
            if c in d.columns
        ]
        if cols:
            st.dataframe(d[cols].head(30), width="stretch")
        st.caption("Candidate effects are screening signals, not measured savings.")
    else:
        st.info("No counterfactual artifact is available.")

with tabs[3]:
    f = ADV / "ai_exp35_intervention_reduction_mapping.csv"
    source = "pipeline output"
    if not f.exists():
        f = EXAMPLES / "intervention_mapping_demo.csv"
        source = "curated public-data example"

    if f.exists():
        d = pd.read_csv(f)
        st.write(f"Intervention mappings: **{len(d)}** · source: **{source}**")
        cols = [
            c
            for c in [
                "building_id",
                "building_type",
                "energy_archetype",
                "cause_candidate",
                "best_measure_family",
                "action_confidence",
                "annual_meter_reading",
                "energy_saving_20%",
                "action_priority",
            ]
            if c in d.columns
        ]
        if cols:
            st.dataframe(d[cols].head(30), width="stretch")
        st.caption(
            "10/20/30% values are scenario assumptions, not measured engineering savings."
        )
    else:
        st.info("No intervention mapping artifact is available.")

with tabs[4]:
    annual = st.number_input(
        "Annual energy reading",
        min_value=0.0,
        max_value=1e9,
        value=1000000.0,
        step=10000.0,
    )
    reduction = st.slider("Assumed reduction (%)", 0, 50, 20) / 100
    ef = st.number_input(
        "Emission factor (kgCO₂e/kWh)",
        min_value=0.0,
        max_value=2.0,
        value=0.5777,
        step=0.01,
    )
    st.metric("Scenario CO₂e", f"{annual * reduction * ef / 1000:,.1f} t")
    st.warning(
        "Conditional scenario only: verify the meter unit and emission factor first."
    )
