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
