# System Architecture

## Layer 1 — Data
Building ID, building type, floor area, site, daily meter readings and weather.

## Layer 2 — Prediction
Historical lag/rolling features are shifted so future observations cannot enter the feature set. ExtraTrees models non-linear relationships.

## Layer 3 — Diagnosis
A 28-day rolling median represents recent expected behavior. A 56-day MAD-derived robust scale reduces sensitivity to extreme readings.

## Layer 4 — Fingerprint
Buildings are characterized by anomaly persistence, recurrence, seasonality, weekend behavior and long-term deviation.

## Layer 5 — Counterfactual
The system estimates what energy use could have looked like without an intervention. Placebo and calibration analysis are used to distinguish model error from potential effects.

## Layer 6 — Intervention
Patterns are mapped to inspection families such as schedule/control audit, occupancy-operation review, HVAC setpoint review and maintenance investigation.

## Layer 7 — Carbon
Energy × emission factor is used only as a parameterized scenario until units and factors are verified.

## Layer 8 — Validation
The final evidence standard is field data with treatment/control groups, before/after measurements, DID and AI counterfactual comparison.