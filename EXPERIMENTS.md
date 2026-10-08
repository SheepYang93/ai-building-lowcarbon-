# Experiment Record

## Prediction
Historical experiments compared tree-based models under rolling time validation. The strongest configurations achieved roughly R² ≈ 0.98 and WAPE ≈ 6–8% in the public dataset, depending on split and subgroup.

## Anomaly detection
A rolling-median + robust-MAD method was developed after rejecting naive actual/predicted ratios that were unstable near zero baselines.

## Robustness
Priority rankings were tested under alternative peer definitions, climate/site normalization, bootstrap resampling and multi-indicator screening.

## Counterfactual
Expected-use screening is treated as a prioritization signal rather than a causal effect. Placebo/falsification checks are reserved for field-validation data. High apparent effects near low baselines should be treated as unstable rather than automatically accepted.

## Intervention
Intervention families were mapped from temporal/behavioral fingerprints. Their effect rates are scenario assumptions, not measured engineering savings.

## Carbon
Carbon scenarios use an emission factor only conditionally. The public derivative's meter-unit lineage must be verified before treating readings as kWh.

## Field validation
The validation prototype now supports treatment/control DID, a simple pre-trend slope diagnostic, a pre-period placebo DID and clustered bootstrap intervals. These are evidence safeguards, not proof of causality; weather, seasonality, autocorrelation and intervention compliance still require study-specific treatment.