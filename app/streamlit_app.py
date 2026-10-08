import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from app.economics import scenario
    from app.emission_factors import EMISSION_FACTORS
    from app.validation import did_effect, pretrend_check
except ImportError:
    from economics import scenario
    from emission_factors import EMISSION_FACTORS
    from validation import did_effect, pretrend_check

st.set_page_config(page_title="AI Building Low-carbon Intelligence", page_icon="🏢", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ADV = OUT / "advanced"
EXAMPLES = ROOT / "examples"


def load_metrics():
    metrics_path = OUT / "metrics.json"
    demo_metrics_path = EXAMPLES / "metrics_2017_public_data.json"
    path = metrics_path if metrics_path.exists() else demo_metrics_path
    return json.loads(path.read_text(encoding="utf-8"))


def load_uploaded_building(uploaded_file):
    df = pd.read_csv(uploaded_file)
    original_columns = list(df.columns)
    aliases = {
        "timestamp": ["timestamp", "datetime", "date", "time"],
        "energy": [
            "electricity_kwh",
            "energy_kwh",
            "meter_reading",
            "energy",
            "electricity",
        ],
        "building_id": ["building_id", "building", "id"],
        "building_type": ["building_type", "sub_primaryspaceusage", "type"],
        "area_m2": ["area_m2", "sqm", "area"],
        "outdoor_temperature": [
            "outdoor_temperature",
            "airTemperature",
            "temperature",
            "temp",
        ],
    }

    rename = {}
    lower = {str(c).strip().lower(): c for c in df.columns}
    for target, candidates in aliases.items():
        for candidate in candidates:
            source = lower.get(candidate.lower())
            if source is not None:
                rename[source] = target
                break

    df = df.rename(columns=rename)

    if "timestamp" not in df.columns:
        raise ValueError(
            "Missing time column. Use timestamp, datetime, date, or time."
        )
    if "energy" not in df.columns:
        raise ValueError(
            "Missing energy column. Use electricity_kwh, energy_kwh, "
            "meter_reading, energy, or electricity."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df = df.dropna(subset=["timestamp", "energy"]).copy()
    df = df[df["energy"] >= 0].copy()

    if df.empty:
        raise ValueError("No valid rows remain after parsing time and energy.")

    if "building_id" in df.columns:
        building_count = df["building_id"].dropna().astype(str).nunique()
        if building_count > 1:
            raise ValueError(
                "This demo currently diagnoses one building at a time. "
                f"The uploaded file contains {building_count} building_id values."
            )

    df["date"] = df["timestamp"].dt.floor("D")
    daily = df.groupby("date", as_index=False)["energy"].sum()
    daily = daily.sort_values("date").reset_index(drop=True)

    for optional in ["building_id", "building_type", "area_m2"]:
        if optional in df.columns:
            non_null = df[optional].dropna()
            if len(non_null):
                daily[optional] = non_null.iloc[0]

    return daily, original_columns


try:
    from app.diagnostic_core import diagnose_uploaded_data
except ImportError:
    from diagnostic_core import diagnose_uploaded_data




def score_upload_quality(daily):
    """Return a transparent 0-100 data-quality score for the uploaded building series."""
    checks = {}
    checks["timestamp_valid"] = float(daily["date"].notna().mean())
    checks["energy_valid"] = float(daily["energy"].notna().mean())
    checks["energy_nonnegative"] = float((daily["energy"] >= 0).mean())
    checks["no_duplicate_dates"] = float((~daily["date"].duplicated()).mean())
    span_days = (daily["date"].max() - daily["date"].min()).days + 1
    coverage = min(len(daily) / max(span_days, 1), 1.0)
    checks["temporal_coverage"] = coverage
    checks["temperature_available"] = float(
        daily["outdoor_temperature"].notna().mean()
    ) if "outdoor_temperature" in daily.columns else 0.0
    score = round(100 * (
        0.20 * checks["timestamp_valid"]
        + 0.25 * checks["energy_valid"]
        + 0.15 * checks["energy_nonnegative"]
        + 0.15 * checks["no_duplicate_dates"]
        + 0.15 * checks["temporal_coverage"]
        + 0.10 * checks["temperature_available"]
    ), 1)
    return score, checks


def run_uploaded_ai_screening(daily):
    """Train a small chronological ExtraTrees model when upload fields are sufficient."""
    required = {"building_id", "area_m2", "outdoor_temperature"}
    if not required.issubset(daily.columns) or len(daily) < 60:
        return None, "AI prediction requires building_id, area_m2, outdoor_temperature and at least 60 daily records."
    try:
        from src.pipeline import features, train_model
        d = daily.copy().rename(columns={
            "area_m2": "sqm",
            "outdoor_temperature": "airTemperature",
            "energy": "meter_reading",
        })
        d["meter"] = "electricity"
        d["sub_primaryspaceusage"] = d.get("building_type", "Uploaded building")
        d["site_id"] = "uploaded"
        d["date"] = pd.to_datetime(d["date"])
        d["sqm"] = pd.to_numeric(d["sqm"], errors="coerce")
        d["airTemperature"] = pd.to_numeric(d["airTemperature"], errors="coerce")
        d["meter_reading"] = pd.to_numeric(d["meter_reading"], errors="coerce")
        for weather_col in ["cloudCoverage", "dewTemperature", "windSpeed"]:
            if weather_col not in d.columns:
                d[weather_col] = 0.0
        d = d.dropna(subset=["building_id", "date", "sqm", "airTemperature", "meter_reading"])
        d = d[d["sqm"] > 0].sort_values(["building_id", "date"])
        if len(d) < 60 or d["date"].nunique() < 60:
            return None, "Not enough valid daily observations for chronological AI screening."
        featured, feature_cols = features(d)
        model, _, test, pred, cutoff, metrics_ai = train_model(featured, feature_cols, cap=50000)
        out = test[["building_id", "date", "meter_reading"]].copy()
        out["predicted_energy"] = pred
        out["residual"] = out["meter_reading"] - out["predicted_energy"]
        out["relative_gap"] = out["residual"] / out["predicted_energy"].abs().clip(lower=1e-9)
        out["residual_baseline"] = out["residual"].shift(1).rolling(14, min_periods=7).median()
        out["residual_mad"] = out["residual"].shift(1).rolling(14, min_periods=7).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        scale = (1.4826 * out["residual_mad"]).clip(lower=1e-9)
        out["residual_robust_z"] = (out["residual"] - out["residual_baseline"]) / scale
        out["ai_anomaly"] = (
            out["residual_baseline"].notna()
            & (out["residual_robust_z"] >= 3.0)
        ).astype(int)
        out["anomaly_confidence"] = np.clip(
            (out["residual_robust_z"] - 2.0) / 3.0, 0.0, 1.0
        )
        return {"data": out, "metrics": metrics_ai, "cutoff": cutoff, "model": model}, None
    except Exception as exc:
        return None, f"AI screening unavailable: {exc}"

def build_report(summary, d, metadata):
    peak = d.loc[d["anomaly"], ["date", "energy", "baseline_28d", "robust_z"]].copy()
    peak = peak.sort_values("robust_z", ascending=False).head(10)

    lines = [
        "# AI Building Energy Diagnosis Report",
        "",
        "## 1. Building",
        f"- Building ID: {metadata.get('building_id', 'not provided')}",
        f"- Building type: {metadata.get('building_type', 'not provided')}",
        f"- Area (m²): {metadata.get('area_m2', 'not provided')}",
        f"- Analysis period: {summary['start']} to {summary['end']}",
        "",
        "## 2. Data quality",
        f"- Valid daily records: {summary['records']}",
        "",
        "## 3. Diagnostic result",
        f"- Robust anomaly days: {summary['anomaly_days']}",
        f"- Anomaly share: {summary['anomaly_share'] * 100:.1f}%",
        f"- Energy fingerprint: {summary['fingerprint']}",
        f"- Recent-vs-early median trend: {summary['trend_pct']:.1f}%",
        (
            f"- Weekend/weekday median ratio: {summary['weekend_ratio']:.2f}"
            if not pd.isna(summary["weekend_ratio"])
            else "- Weekend/weekday median ratio: unavailable"
        ),
        "",
        "## 4. Counterfactual screening signal",
        "- Baseline: trailing 28-day median using past observations only",
        f"- Potential reduction signal: {summary['potential_pct']:.1f}%",
        "- This is a model-based screening signal, not measured savings.",
        "",
        "## 5. Suggested intervention",
        f"- {summary['intervention']}",
        "- Verify operating schedules, HVAC settings and equipment status before intervention.",
        "",
        "## 6. Highest anomaly dates",
    ]

    if len(peak):
        lines.extend([
            "",
            "| Date | Energy | Past baseline | Robust z |",
            "|---|---:|---:|---:|",
        ])
        for _, row in peak.iterrows():
            lines.append(
                f"| {row['date'].date()} | {row['energy']:.2f} | "
                f"{row['baseline_28d']:.2f} | {row['robust_z']:.2f} |"
            )
    else:
        lines.append("- No robust anomaly dates detected.")

    lines.extend([
        "",
        "## 7. Evidence boundary",
        "- Uploaded-data screening uses a trailing robust baseline; it is not the same as the validated ExtraTrees model reported for the public dataset.",
        "- Counterfactual and intervention values are screening hypotheses and must not be presented as realized energy or carbon savings.",
        "- For real campus deployment, validate with treatment/control, Difference-in-Differences and placebo tests.",
    ])
    return "\n".join(lines)


metrics = load_metrics()
anomaly_path = OUT / "anomaly_buildings.csv"
an = pd.read_csv(anomaly_path) if anomaly_path.exists() else pd.DataFrame()

st.markdown("""
<style>
.block-container {max-width:1500px; padding-top:2rem;}
[data-testid="stMetric"] {background:#fff;border:1px solid #E2EBE8;border-radius:16px;padding:12px 16px;box-shadow:0 3px 16px rgba(23,59,54,.05)}
[data-testid="stMetricLabel"] {color:#617873}
[data-testid="stMetricValue"] {color:#173B36}
.hero {padding:2.2rem 2.4rem;border-radius:24px;margin-bottom:1.3rem;background:linear-gradient(135deg,#0F766E,#155E75);color:#fff}
.hero h1 {font-size:2.4rem;margin:0 0 .45rem;color:#fff}
.hero p {font-size:1.05rem;margin:0;color:#E7FFFA}
.card {background:#fff;border:1px solid #E2EBE8;border-radius:18px;padding:1.1rem 1.2rem;min-height:125px}
.badge {display:inline-block;padding:.22rem .58rem;border-radius:999px;background:#E6F4F0;color:#0F766E;font-size:.76rem;font-weight:700}
.muted {color:#617873;font-size:.9rem}
.warning-card {background:#FFF9E9;border:1px solid #F0DFAB;border-radius:14px;padding:.85rem 1rem;color:#66501A}
.section-title {font-size:1.35rem;font-weight:750;color:#173B36;margin:.35rem 0 .25rem}
.section-subtitle {color:#617873;margin-bottom:1rem}
.insight-card {background:#F2F8F6;border-left:4px solid #0F766E;border-radius:12px;padding:1rem 1.1rem;margin:.5rem 0}
.data-status {background:#F7FAF9;border:1px solid #E2EBE8;border-radius:12px;padding:.8rem 1rem;margin:.7rem 0 1rem;color:#315B53}
.chip {display:inline-block;margin:.28rem .35rem .1rem 0;padding:.42rem .68rem;border:1px solid #D8E8E4;border-radius:999px;background:#fff;color:#315B53;font-size:.82rem;font-weight:650}
.hero-grid {display:grid;grid-template-columns:1.5fr .8fr;gap:1rem;align-items:end}
.hero-kicker {font-size:.72rem;letter-spacing:.14em;font-weight:800;opacity:.82}
.hero-side {background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:1rem 1.1rem}
.hero-side .small {font-size:.76rem;opacity:.78}
.hero-side .big {font-size:1.45rem;font-weight:800;margin-top:.15rem}
.section-rule {height:1px;background:#E5EFEC;margin:1.2rem 0}
[data-baseweb="tab-list"] {gap:.3rem;background:#F7FAF9;border:1px solid #E2EBE8;border-radius:14px;padding:.3rem}
[data-baseweb="tab"] {border-radius:10px;font-weight:650;color:#55706A}
[data-testid="stDataFrame"] {border:1px solid #E2EBE8;border-radius:14px;overflow:hidden}
@media (max-width:900px){.hero-grid{grid-template-columns:1fr}.hero{padding:1.5rem}.hero h1{font-size:1.9rem}}
</style>
<div class="hero">
<div class="hero-grid">
  <div>
    <div class="hero-kicker">AI · ENERGY · CARBON INTELLIGENCE</div>
    <h1>Building Energy & Low-carbon Intelligence</h1>
    <p>从能源数据识别异常、理解建筑行为，并把模型结果转化为成本与碳排放情景。</p>
  </div>
  <div class="hero-side">
    <div class="small">DECISION SUPPORT PROTOTYPE</div>
    <div class="big">Data → Diagnosis → Action</div>
    <div class="small">Model evidence ≠ measured savings</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Overview",
    "Upload & Diagnose",
    "Building Diagnosis",
    "Counterfactual",
    "Intervention",
    "Carbon",
    "Report",
    "Field Validation",
])

with tabs[0]:
    st.markdown('<div class="section-title">Portfolio Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">AI 驱动的建筑能源异常诊断与低碳决策支持 Demo。</div>', unsafe_allow_html=True)

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

    st.markdown("### What this system does")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            '''
<div class="insight-card">
<b>From energy data to operational decisions</b><br>
系统不是只做“能耗预测”，而是把预测、异常识别、反事实筛选、干预建议和碳情景连接起来。
</div>
''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
<span class="chip">01 · Predict</span>
<span class="chip">02 · Detect anomalies</span>
<span class="chip">03 · Build fingerprint</span>
<span class="chip">04 · Counterfactual</span>
<span class="chip">05 · Intervention</span>
<span class="chip">06 · Carbon scenario</span>
''',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Decision chain**")
        st.markdown(
            '''
**Energy data**  
↓  
**AI / robust statistical diagnosis**  
↓  
**Operational cause candidates**  
↓  
**Intervention screening**  
↓  
**Cost & CO₂e scenario**  
↓  
**Field validation**
''')
    st.markdown("### Evidence at a glance")
    e1, e2, e3 = st.columns(3)
    e1.markdown(
        '<div class="insight-card"><b>Model evidence</b><br>Public-data prediction and strict time-series validation.</div>',
        unsafe_allow_html=True,
    )
    e2.markdown(
        '<div class="insight-card"><b>Decision evidence</b><br>Robust anomaly detection, energy fingerprints and intervention mapping.</div>',
        unsafe_allow_html=True,
    )
    e3.markdown(
        '<div class="insight-card"><b>Validation design</b><br>Treatment / control, DID and placebo testing workflow.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Demo boundary: public-data model results and uploaded-data screening are shown separately. "
        "Scenario CO₂e is conditional, not measured emissions reduction."
    )

with tabs[1]:
    st.markdown('<div class="section-title">Upload & Diagnose</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">把一份建筑能耗 CSV 转化为 AI expected-use + robust baseline 双证据诊断；字段不足时自动回退。</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="data-status"><b>Required</b> 时间列 + 能耗列 &nbsp; · &nbsp; <b>AI-ready</b> building_id + area_m2 + outdoor_temperature + ≥60 days &nbsp; · &nbsp; <b>Scope</b> 单建筑 CSV</div>',
        unsafe_allow_html=True,
    )
    st.code(
        "timestamp,electricity_kwh,building_id,building_type,area_m2\n"
        "2026-01-01 00:00,120.5,B001,Classroom,5000\n"
        "2026-01-01 01:00,115.2,B001,Classroom,5000"
    )
    sample_path = EXAMPLES / "sample_building_energy.csv"
    if sample_path.exists():
        st.download_button(
            "Download sample CSV",
            data=sample_path.read_text(encoding="utf-8"),
            file_name="sample_building_energy.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("CSV file", type=["csv"], key="building_upload")

    if uploaded is not None:
        try:
            daily, original_columns = load_uploaded_building(uploaded)
            diagnosed, summary = diagnose_uploaded_data(daily)

            metadata = {
                "building_id": (
                    daily["building_id"].iloc[0]
                    if "building_id" in daily.columns
                    else "not provided"
                ),
                "building_type": (
                    daily["building_type"].iloc[0]
                    if "building_type" in daily.columns
                    else "not provided"
                ),
                "area_m2": (
                    daily["area_m2"].iloc[0]
                    if "area_m2" in daily.columns
                    else "not provided"
                ),
            }

            st.session_state["upload_daily"] = diagnosed
            st.session_state["upload_summary"] = summary
            st.session_state["upload_metadata"] = metadata
            st.session_state["upload_report"] = build_report(
                summary, diagnosed, metadata
            )

            a, b, c, d = st.columns(4)
            a.metric("Daily records", summary["records"])
            b.metric("Anomaly days", summary["anomaly_days"])
            c.metric("Anomaly share", f'{summary["anomaly_share"] * 100:.1f}%')
            d.metric("Potential signal", f'{summary["potential_pct"]:.1f}%')

            if summary["records"] < 35:
                st.warning(
                    "The uploaded period is shorter than 35 days. "
                    "The 28-day robust baseline will have limited context; "
                    "use 8–12 weeks for a stronger field validation baseline."
                )

            quality_score, quality_checks = score_upload_quality(daily)
            q1, q2 = st.columns([1, 3])
            q1.metric("Data quality", f"{quality_score:.0f}/100")
            q2.caption("Quality score covers timestamp/energy validity, non-negative readings, duplicate dates, temporal coverage and temperature availability. It is a screening score, not a certification.")
            
            ai_result, ai_message = run_uploaded_ai_screening(daily)
            if ai_result is not None:
                st.session_state["upload_ai"] = ai_result
                st.success(
                    f"AI screening available: chronological ExtraTrees trained on the uploaded history "
                    f"and evaluated on the holdout period (R²={ai_result['metrics']['r2']:.3f}, "
                    f"WAPE={ai_result['metrics']['wape'] * 100:.1f}%)."
                )
                ai1, ai2, ai3 = st.columns(3)
                ai1.metric("AI test R²", f"{ai_result['metrics']['r2']:.3f}")
                ai2.metric("AI test WAPE", f"{ai_result['metrics']['wape'] * 100:.1f}%")
                ai3.metric("AI anomaly days", int(ai_result["data"]["ai_anomaly"].sum()))
                anomaly_score = ai_result["data"]["anomaly_confidence"].mean()
                st.metric("Mean anomaly confidence", f"{anomaly_score * 100:.0f}%")
                st.caption("Anomaly score uses a past-only 14-day residual median/MAD. It is a screening score, not a probability of fault or verified savings.")
                ai_plot = ai_result["data"].set_index("date")[["meter_reading", "predicted_energy"]].rename(
                    columns={"meter_reading": "Actual", "predicted_energy": "Expected"}
                )
                st.markdown("**Actual vs expected energy**")
                st.line_chart(ai_plot)
                st.caption("Uploaded-data AI screening is retrained on the uploaded history; it is not the public-data benchmark model.")
            else:
                st.info(ai_message + " The robust past-only baseline remains available.")

            st.markdown('<div class="section-title">Building health</div>', unsafe_allow_html=True)
            card1, card2, card3 = st.columns(3)
            card1.metric("Status", summary["fingerprint"])
            card2.metric("Anomaly level", f'{summary["anomaly_share"] * 100:.1f}% of valid days')
            card3.metric("Trend", f'{summary["trend_pct"]:+.1f}%')

            weekend_ratio_label = (
                f"{summary['weekend_ratio']:.2f}"
                if not pd.isna(summary["weekend_ratio"])
                else "unavailable"
            )
            st.markdown(
                f"""
**Building:** {metadata["building_id"]}

**Type:** {metadata["building_type"]} · **Area:** {metadata["area_m2"]} m²

**Main evidence**
- {summary["anomaly_days"]} robust anomaly days detected
- Weekend/weekday median ratio: {weekend_ratio_label}
- Screening signal: {summary["potential_pct"]:.1f}%

**Suggested first checks**
- {summary["intervention"]}
- Verify HVAC operating schedule and setpoints
- Check non-occupancy equipment operation
- Compare the anomaly dates with occupancy/calendar records

> **Evidence boundary:** this card is a screening result. The potential signal is not measured savings and should be validated with treatment/control, Difference-in-Differences and placebo analysis.
"""
            )

            st.write({
                "columns_received": original_columns,
                "period": f"{summary['start']} → {summary['end']}",
                "fingerprint": summary["fingerprint"],
                "suggested_intervention": summary["intervention"],
            })

            st.markdown('<div class="section-title">Energy fingerprint</div>', unsafe_allow_html=True)
            fp1, fp2 = st.columns(2)
            with fp1:
                st.line_chart(diagnosed.set_index("date")[["energy", "baseline_28d"]])
            with fp2:
                weekly = diagnosed.set_index("date")["energy"].resample("W").sum()
                st.bar_chart(weekly)

            anomaly_view = diagnosed[diagnosed["anomaly"]].copy()
            if len(anomaly_view):
                st.subheader("Detected anomaly dates")
                st.dataframe(
                    anomaly_view[
                        ["date", "energy", "baseline_28d", "ratio_to_baseline", "robust_z"]
                    ].tail(50),
                    width="stretch",
                )
            else:
                st.info("No robust anomaly dates were detected in the uploaded period.")

        except Exception as exc:
            st.error(f"Upload or diagnosis failed: {exc}")

with tabs[2]:
    st.markdown('<div class="section-title">Building Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">从异常频率、优先级和建筑类型快速定位值得进一步检查的对象。</div>', unsafe_allow_html=True)
    if len(an):
        q = an.copy()
        q["label"] = q.building_id.astype(str) + " · " + q.sub_primaryspaceusage.astype(str)
        label = st.selectbox("Select building", q.label.head(200).tolist())
        r = q[q.label.eq(label)].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Priority", f"{r.priority_score:.1f}")
        b.metric("Anomaly days", int(r.anomaly_days))
        c.metric("Anomaly share", f"{r.anomaly_share * 100:.1f}%")
        d.metric("Area", f"{float(r.sqm):,.0f} m²")
        st.markdown(
            f'<div class="insight-card"><b>Diagnosis summary</b><br>{r.sub_primaryspaceusage} building · priority {float(r.priority_score):.1f} · {int(r.anomaly_days)} anomaly days. Recommended next step: inspect operating schedule, HVAC settings and equipment status.</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Building profile**")
            st.dataframe(pd.DataFrame([{
                "Building ID": r.building_id,
                "Type": r.sub_primaryspaceusage,
                "Area (m²)": round(float(r.sqm), 1),
                "Site": r.site_id,
            }]), width="stretch", hide_index=True)
        with right:
            st.markdown("**Priority interpretation**")
            priority = float(r.priority_score)
            if priority >= 75:
                st.warning("High priority · recommend operational review first.")
            elif priority >= 50:
                st.info("Medium priority · monitor and compare with operating schedule.")
            else:
                st.success("Lower priority · keep under routine monitoring.")
    else:
        st.info("Live building-diagnosis output is not bundled in the public demo. Use Upload & Diagnose for your own building data.")

with tabs[3]:
    st.markdown('<div class="section-title">Expected-use / Counterfactual Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">用历史正常行为构建 expected-use baseline，筛选值得进一步验证的潜在节能空间。</div>', unsafe_allow_html=True)

    cf = ADV / "ai_exp45_robust_candidates.csv"
    source = "pipeline output"
    if not cf.exists():
        cf = EXAMPLES / "counterfactual_candidates_demo.csv"
        source = "curated public-data example"

    if cf.exists():
        data = pd.read_csv(cf)
        st.markdown(
            f'<div class="data-status"><b>{len(data)}</b> candidate records &nbsp; · &nbsp; source: <b>{source}</b></div>',
            unsafe_allow_html=True,
        )
        cols = [c for c in [
            "building_id", "building_type", "protected_calibrated_pct",
            "robust_low_pct", "robust_high_pct", "robust_confidence", "robust_class"
        ] if c in data.columns]

        if len(data):
            row = data.iloc[0]
            k1, k2, k3 = st.columns(3)
            if "protected_calibrated_pct" in data:
                k1.metric("Median screening effect", f'{data["protected_calibrated_pct"].median():.1f}%')
            if "robust_confidence" in data:
                k2.metric("Median confidence", f'{data["robust_confidence"].median():.2f}')
            if "robust_class" in data:
                k3.metric("Candidate class", str(row["robust_class"]))

            st.markdown(
                '<div class="insight-card"><b>How to read this</b><br>'
                'Expected-use / counterfactual screening estimates a plausible comparison point from historical behavior. '
                'It helps prioritize buildings for investigation; it does not establish realized savings.</div>',
                unsafe_allow_html=True,
            )

        if cols:
            st.markdown("**Candidate details**")
            st.dataframe(data[cols].head(30), width="stretch", hide_index=True)
        st.caption("Candidate effects are screening signals, not measured savings.")
    else:
        st.info("No counterfactual artifact is available.")

with tabs[4]:
    st.markdown('<div class="section-title">Intervention Mapping</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">把异常模式映射到下一步应该检查的设备、运行策略和管理动作。</div>', unsafe_allow_html=True)

    f = ADV / "ai_exp35_intervention_reduction_mapping.csv"
    source = "pipeline output"
    if not f.exists():
        f = EXAMPLES / "intervention_mapping_demo.csv"
        source = "curated public-data example"

    if f.exists():
        data = pd.read_csv(f)
        st.markdown(
            f'<div class="data-status"><b>{len(data)}</b> intervention mappings &nbsp; · &nbsp; source: <b>{source}</b></div>',
            unsafe_allow_html=True,
        )
        cols = [c for c in [
            "building_id", "building_type", "energy_archetype", "cause_candidate",
            "best_measure_family", "action_confidence", "annual_meter_reading",
            "energy_saving_20%", "action_priority"
        ] if c in data.columns]

        if len(data):
            priority_col = "action_priority" if "action_priority" in data.columns else None
            confidence_col = "action_confidence" if "action_confidence" in data.columns else None
            k1, k2, k3 = st.columns(3)
            if priority_col:
                k1.metric("Top priority", str(data[priority_col].iloc[0]))
            if confidence_col:
                k2.metric("Top confidence", f'{float(data[confidence_col].iloc[0]):.2f}')
            if "energy_saving_20%" in data.columns:
                k3.metric("Scenario @20%", f'{data["energy_saving_20%"].median():.1f}%')

            st.markdown("**Recommended action map**")
            for _, row in data.head(5).iterrows():
                building = row.get("building_id", "Building")
                cause = row.get("cause_candidate", "Operational anomaly")
                measure = row.get("best_measure_family", "Further investigation")
                priority = row.get("action_priority", "—")
                confidence = row.get("action_confidence", "—")
                st.markdown(
                    f'<div class="insight-card"><b>{building}</b> · priority <b>{priority}</b><br>'
                    f'<b>Possible cause:</b> {cause}<br>'
                    f'<b>Recommended measure:</b> {measure}<br>'
                    f'<span class="muted">Action confidence: {confidence}</span></div>',
                    unsafe_allow_html=True,
                )

        if cols:
            with st.expander("View structured intervention data"):
                st.dataframe(data[cols].head(30), width="stretch", hide_index=True)

        st.caption("10/20/30% values are scenario assumptions, not measured engineering savings.")
    else:
        st.info("No intervention mapping artifact is available.")

with tabs[5]:
    st.markdown('<div class="section-title">Energy · Cost · Carbon scenario</div>', unsafe_allow_html=True)
    st.caption("Location-based Scope 2 screening scenario. Values are estimates, not measured savings.")
    left, right = st.columns([1.0, 1.45])
    with left:
        annual = st.number_input("Annual energy (kWh)", min_value=0.0, max_value=1e9, value=1000000.0, step=10000.0)
        reduction = st.slider("Assumed reduction", 0, 50, 20) / 100
        tariff = st.number_input("Electricity tariff (¥/kWh)", min_value=0.0, max_value=10.0, value=0.80, step=0.05)
        factor_key = st.selectbox("Emission factor", list(EMISSION_FACTORS))
        factor = EMISSION_FACTORS[factor_key]["value"]
        st.caption(f"Source: {EMISSION_FACTORS[factor_key]['source']} · {EMISSION_FACTORS[factor_key]['unit']}")
    with right:
        result = scenario(annual, reduction, tariff, factor)
        a, b, c = st.columns(3)
        a.metric("Avoided energy", f'{result["avoided_energy_kwh"]:,.0f} kWh')
        b.metric("Estimated energy-charge saving", f'¥{result["estimated_energy_charge_saving"]:,.0f}')
        c.metric("Avoided CO₂e", f'{result["avoided_co2e_t"]:,.1f} t')
        st.markdown('<div class="warning-card"><b>SCENARIO ONLY</b><br>Verify meter units, reporting boundary, electricity tariff and emission-factor year/geography before real decisions. Cost output is a simplified energy-charge estimate; CO₂ is a location-based Scope 2 screening result, not a measured reduction.</div>', unsafe_allow_html=True)

        scenario_rows = []
        for pct in range(0, 51, 5):
            sr = scenario(annual, pct / 100, tariff, factor)
            scenario_rows.append({
                "Reduction": pct,
                "Energy-charge saving (¥)": sr["estimated_energy_charge_saving"],
                "Avoided CO₂e (t)": sr["avoided_co2e_t"],
            })
        scenario_df = pd.DataFrame(scenario_rows).set_index("Reduction")
        st.markdown('<div class="section-title">Scenario sensitivity</div>', unsafe_allow_html=True)
        st.caption("同一能耗基线下，不同假设节能率对应的成本与碳减排情景。")
        st.line_chart(scenario_df[["Energy-charge saving (¥)", "Avoided CO₂e (t)"]], height=300)

with tabs[6]:
    st.markdown('<div class="section-title">Diagnostic Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">把最近一次上传数据的诊断结果整理成可复用的项目输出。</div>', unsafe_allow_html=True)
    report = st.session_state.get("upload_report")
    if report:
        st.markdown(
            "The report below was generated from the most recent uploaded dataset."
        )
        st.download_button(
            "Download Markdown report",
            data=report,
            file_name="building_energy_diagnosis_report.md",
            mime="text/markdown",
        )
        st.markdown('<div class="insight-card"><b>Evidence boundary</b><br>报告中的 screening signal、情景节能率和 CO₂e 均不等同于实测节能或实测减排；正式应用前需要现场验证。</div>', unsafe_allow_html=True)
        st.text_area("Report preview", report, height=600)
    else:
        st.info(
            "Upload a building CSV in Upload & Diagnose first. "
            "The system will generate a diagnosis report automatically."
        )


with tabs[7]:
    st.markdown('<div class="section-title">Field Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">用 Treatment / Control + DID 把模型筛选结果推进到现场证据。</div>', unsafe_allow_html=True)
    st.markdown('<div class="data-status"><b>Evidence path</b> Baseline → intervention → treatment/control → DID → placebo / pre-trend checks</div>', unsafe_allow_html=True)
    st.caption("This is a basic DID calculator for study-design screening; it does not yet run automated pre-trend or placebo tests.")
    st.write(
        "Upload a validation CSV with one row per building-day. Required columns: "
        "date, energy, building_id, group, intervention_date. "
        "Use group=treated for buildings receiving the intervention and group=control "
        "for comparable buildings without the intervention."
    )
    st.code(
        "date,energy,building_id,group,intervention_date\n"
        "2026-03-01,1200,B001,treated,2026-04-01\n"
        "2026-03-01,1180,B002,control,2026-04-01"
    )
    template = (
        "date,energy,building_id,group,intervention_date\n"
        "2026-03-01,1200,B001,treated,2026-04-01\n"
        "2026-03-01,1180,B002,control,2026-04-01\n"
    )
    st.download_button(
        "Download validation template",
        data=template,
        file_name="field_validation_template.csv",
        mime="text/csv",
    )

    validation_file = st.file_uploader(
        "Validation CSV", type=["csv"], key="validation_upload"
    )

    if validation_file is not None:
        try:
            v = pd.read_csv(validation_file)
            required = {"date", "energy", "building_id", "group", "intervention_date"}
            missing = required - set(v.columns)
            if missing:
                raise ValueError(
                    "Missing required columns: " + ", ".join(sorted(missing))
                )

            v["date"] = pd.to_datetime(v["date"], errors="coerce")
            v["intervention_date"] = pd.to_datetime(
                v["intervention_date"], errors="coerce"
            )
            v["energy"] = pd.to_numeric(v["energy"], errors="coerce")
            v["group"] = v["group"].astype(str).str.lower().str.strip()
            v = v.dropna(
                subset=["date", "energy", "building_id", "intervention_date"]
            ).copy()
            v = v[v["group"].isin(["treated", "control"])].copy()

            if v.empty:
                raise ValueError("No valid treated/control rows remain.")

            intervention_date = v["intervention_date"].mode().iloc[0]
            if v["intervention_date"].nunique() > 1:
                st.warning("Multiple intervention dates detected; using the most frequent date for this screening run.")
            pre = v["date"] < intervention_date
            post = v["date"] >= intervention_date

            if not pre.any() or not post.any():
                raise ValueError(
                    "The dataset needs observations both before and after the intervention date."
                )

            result = did_effect(v, intervention_date)
            summary = result["summary"]
            treated_change = result["treated_change"]
            control_change = result["control_change"]
            did_abs = result["did_abs"]
            did_pct = result["did_pct"]

            st.success("Validation analysis completed.")
            a, b, c = st.columns(3)
            a.metric("Treated pre → post", f"{treated_change:+.2f}")
            b.metric("Control pre → post", f"{control_change:+.2f}")
            c.metric("DID effect", f"{did_abs:+.2f}")

            st.metric(
                "DID relative to treated baseline",
                f"{did_pct:+.2f}%" if not pd.isna(did_pct) else "unavailable",
            )

            st.dataframe(summary.reset_index(), width="stretch")

            plot_df = (
                v.assign(period=np.where(pre, "Pre", "Post"))
                .groupby(["date", "group"])["energy"]
                .mean()
                .reset_index()
            )
            pivot = plot_df.pivot(index="date", columns="group", values="energy")
            st.subheader("Treatment / control energy trend")
            st.line_chart(pivot)

            trend = pretrend_check(v, intervention_date)
            st.markdown("**Pre-trend diagnostic**")
            if trend["status"] == "diagnostic":
                t1, t2, t3 = st.columns(3)
                t1.metric("Treated pre-slope", f'{trend["treated_slope"]:+.4f}/day')
                t2.metric("Control pre-slope", f'{trend["control_slope"]:+.4f}/day')
                t3.metric("Slope gap", f'{trend["slope_gap"]:+.4f}/day')
                st.caption("This is a simple diagnostic of pre-period slopes, not a formal parallel-trends test.")
            else:
                st.info("Not enough pre-period observations to compute a slope diagnostic.")

            st.warning(
                "Interpretation boundary: DID compares the treated group's change with "
                "the control group's change. A negative DID means treated energy fell "
                "more than control energy. This calculator does not automatically test "
                "parallel pre-trends, placebo dates, weather confounding or statistical "
                "significance, so the result should be treated as preliminary evidence."
            )

            st.download_button(
                "Download validation summary",
                data=summary.reset_index().to_csv(index=False),
                file_name="did_validation_summary.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Validation analysis failed: {exc}")
