# System Architecture

## Layer 1 — Data
Building ID, building type, floor area, site, daily meter readings and weather.

## Layer 2 — Prediction
Historical lag/rolling features are shifted so future observations cannot enter the feature set. ExtraTrees models non-linear relationships.

## Layer 3 — Diagnosis
A 28-day rolling median represents recent expected behavior. A 56-day MAD-derived robust scale reduces sensitivity to extreme readings.

## Layer 4 — Fingerprint
Buildings are characterized by anomaly persistence, recurrence, seasonality, weekend behavior and long-term deviation.

## Layer 5 — Expected-use / Counterfactual Screening
The system constructs a past-only expected-use baseline from historical behavior and uses it to screen potential intervention space. This is not treated as a causal counterfactual unless intervention and control data support that interpretation. Placebo/calibration checks are evidence safeguards, not proof of realized savings.

## Layer 6 — Intervention
Patterns are mapped to inspection families such as schedule/control audit, occupancy-operation review, HVAC setpoint review and maintenance investigation.

## Layer 7 — Carbon
Electricity consumption × a versioned electricity CO₂ emission factor produces a location-based Scope 2 screening scenario. Meter units, geography, reporting year and accounting boundary must be verified before formal carbon accounting.

## Layer 8 — Validation
The final evidence standard is field data with treatment/control groups, before/after measurements, DID and AI counterfactual comparison.