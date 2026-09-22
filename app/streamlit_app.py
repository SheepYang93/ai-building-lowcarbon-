import streamlit as st
from pathlib import Path
import pandas as pd,json
st.set_page_config(page_title="AI Building Low-carbon Decision System",layout="wide")
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; ADV=OUT/"advanced"
st.title("AI Building Energy Anomaly & Low-carbon Decision System"); st.caption("Portfolio prototype · prediction → diagnosis → counterfactual → intervention → carbon scenario")
metrics=json.loads((OUT/"metrics.json").read_text()) if (OUT/"metrics.json").exists() else {}; an=pd.read_csv(OUT/"anomaly_buildings.csv") if (OUT/"anomaly_buildings.csv").exists() else pd.DataFrame()
tabs=st.tabs(["Overview","Building Diagnosis","Counterfactual","Intervention","Carbon"])
with tabs[0]:
    c1,c2,c3,c4=st.columns(4); c1.metric("Buildings",metrics.get("buildings","—")); c2.metric("Test R²",f'{metrics["r2"]:.3f}' if "r2" in metrics else "—"); c3.metric("WAPE",f'{metrics["wape"]*100:.1f}%' if "wape" in metrics else "—"); c4.metric("Anomaly candidates",int((an.anomaly_days>0).sum()) if len(an) else "—")
    st.markdown("**Raw data → Feature engineering → ML prediction → Robust anomaly detection → Energy fingerprint → Counterfactual → Intervention matching → Carbon scenario → Field validation**")
    st.info("Public-data results are screening evidence. Scenario CO₂e is not measured emissions, and predicted savings require field validation.")
with tabs[1]:
    if len(an):
        q=an.copy(); q["label"]=q.building_id.astype(str)+" · "+q.sub_primaryspaceusage.astype(str); label=st.selectbox("Building",q.label.head(200).tolist()); r=q[q.label.eq(label)].iloc[0]
        a,b,c=st.columns(3); a.metric("Priority",f"{r.priority_score:.1f}"); b.metric("Anomaly days",int(r.anomaly_days)); c.metric("Anomaly share",f"{r.anomaly_share*100:.1f}%"); st.write({"building_id":r.building_id,"building_type":r.sub_primaryspaceusage,"area_m2":round(float(r.sqm),1),"site_id":r.site_id})
    else: st.warning("Run V3 pipeline first.")
with tabs[2]:
    cf=ADV/"ai_exp45_robust_candidates.csv"
    if cf.exists():
        d=pd.read_csv(cf); st.write(f"Robust candidate records available: **{len(d)}**"); show=[c for c in ["building_id","building_type","central_effect_pct","robust_low_pct","robust_high_pct","confidence"] if c in d.columns]; st.dataframe(d[show].head(30),use_container_width=True) if show else None
    else: st.info("Add the counterfactual candidate artifact to outputs/advanced.")
with tabs[3]:
    f=ADV/"ai_exp35_intervention_reduction_mapping.csv"
    if f.exists():
        d=pd.read_csv(f); st.write("Intervention families are scenario mappings, not measured engineering savings."); cols=[c for c in ["building_id","fingerprint","intervention_family","annual_reading","saving_20pct"] if c in d.columns]; st.dataframe(d[cols].head(30),use_container_width=True) if cols else None
    else: st.info("Add the intervention mapping artifact to outputs/advanced.")
with tabs[4]:
    annual=st.number_input("Annual energy reading",min_value=0.0,max_value=1e9,value=1000000.0,step=10000.0); reduction=st.slider("Assumed reduction (%)",0,50,20)/100; ef=st.number_input("Emission factor (kgCO₂e/kWh)",min_value=0.0,max_value=2.0,value=.5777,step=.01); st.metric("Scenario CO₂e",f"{annual*reduction*ef/1000:,.1f} t"); st.warning("Conditional scenario only: verify the meter unit and emission factor first.")
