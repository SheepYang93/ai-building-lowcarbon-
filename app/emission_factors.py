"""Versioned electricity emission factors for building Scope 2 scenarios.

The default factors are official Chinese electricity CO2 emission factors
published by the Ministry of Ecology and Environment and National Bureau of
Statistics. They are intended for location-based screening scenarios.

For formal inventories, select the factor matching the reporting year,
geography and accounting boundary. Market-based treatment of traceable
non-fossil electricity requires separate evidence and should not be mixed
with the location-based factor.
"""

EMISSION_FACTORS = {
    "China national · 2023 · location-based": {
        "value": 0.5306,
        "unit": "kgCO2/kWh",
        "source": "MEE + NBS, 2023 national average electricity CO2 emission factor",
        "year": 2023,
        "region": "China",
        "scope": "Scope 2 · location-based screening",
    },
    "Guangdong · 2023 · location-based": {
        "value": 0.4419,
        "unit": "kgCO2/kWh",
        "source": "MEE + NBS, 2023 Guangdong average electricity CO2 emission factor",
        "year": 2023,
        "region": "Guangdong",
        "scope": "Scope 2 · location-based screening",
    },
}


def get_factor(key: str = "China national · 2023 · location-based") -> float:
    return float(EMISSION_FACTORS[key]["value"])
