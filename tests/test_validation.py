import pandas as pd
import pytest
from app.validation import did_effect, pretrend_check

def test_did_effect():
    v=pd.DataFrame({
        "date":pd.to_datetime(["2026-03-01","2026-03-01","2026-04-01","2026-04-01"]),
        "energy":[100.0,100.0,80.0,95.0],
        "group":["treated","control","treated","control"],
    })
    out=did_effect(v,"2026-04-01")
    assert out["did_abs"] == -15
    assert out["did_pct"] == -15

def test_did_requires_both_groups():
    v=pd.DataFrame({
        "date":pd.to_datetime(["2026-03-01","2026-04-01"]),
        "energy":[100.0,90.0],
        "group":["treated","treated"],
    })
    with pytest.raises(ValueError):
        did_effect(v,"2026-04-01")


def test_pretrend_check_returns_slope_diagnostic():
    v=pd.DataFrame({
        "date":pd.to_datetime(["2026-03-01","2026-03-02","2026-03-03","2026-03-01","2026-03-02","2026-03-03"]),
        "energy":[100.0,101.0,102.0,100.0,100.0,100.0],
        "group":["treated","treated","treated","control","control","control"],
    })
    out=pretrend_check(v,"2026-04-01")
    assert out["status"] == "diagnostic"
    assert out["treated_slope"] > out["control_slope"]



def test_placebo_effect_uses_only_pre_intervention_observations():
    from app.validation import placebo_effect

    dates = pd.date_range("2026-03-01", periods=6)
    rows = []
    for group, values in [
        ("treated", [100, 101, 102, 103, 80, 81]),
        ("control", [100, 100, 100, 100, 100, 100]),
    ]:
        for i, value in enumerate(values):
            rows.append({
                "date": dates[i],
                "energy": value,
                "group": group,
                "building_id": f"{group}_1",
            })
    frame = pd.DataFrame(rows)
    pre_only = frame[frame["date"] < pd.Timestamp("2026-03-05")]

    result = placebo_effect(pre_only, "2026-03-03")

    assert result["did_abs"] == 2.0


def test_clustered_bootstrap_did_returns_interval_for_multiple_buildings():
    from app.validation import bootstrap_did

    dates = pd.date_range("2026-03-01", periods=6)
    rows = []
    for group, ids in [("treated", ["t1", "t2"]), ("control", ["c1", "c2"])]:
        for building_id in ids:
            offset = 0 if building_id.endswith("1") else 2
            for i, date in enumerate(dates):
                energy = 100 + offset + i
                if group == "treated" and date >= pd.Timestamp("2026-03-04"):
                    energy -= 10
                rows.append({
                    "date": date,
                    "energy": energy,
                    "group": group,
                    "building_id": building_id,
                })
    result = bootstrap_did(pd.DataFrame(rows), "2026-03-04", n_boot=40, seed=7)

    assert result["status"] == "ok"
    assert result["n"] > 0
    assert result["ci_low"] <= result["ci_high"]
