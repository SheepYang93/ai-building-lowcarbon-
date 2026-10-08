"""Prediction range helper used by the screening UI."""
import numpy as np

def residual_radius(y_true, y_pred, level=0.90):
    r=np.abs(np.asarray(y_true,dtype=float)-np.asarray(y_pred,dtype=float))
    r=r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.quantile(r, level))

def add_range(frame, y_true, y_pred, level=0.90):
    out=frame.copy()
    radius=residual_radius(y_true,y_pred,level)
    out["expected_low"]=out["predicted_energy"]-radius
    out["expected_high"]=out["predicted_energy"]+radius
    out["outside_expected_range"]=((out["meter_reading"]<out["expected_low"])|(out["meter_reading"]>out["expected_high"])).astype(int)
    return out,radius
