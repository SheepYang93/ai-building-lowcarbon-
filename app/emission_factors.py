"""Versioned emission-factor inputs for scenario calculations.

These are configuration examples only. Replace them with factors appropriate
to the target region, accounting boundary and reporting year before deployment.
"""

EMISSION_FACTORS = {
    "grid_electricity_example": {
        "value": 0.5777,
        "unit": "kgCO2e/kWh",
        "source": "demo configuration — verify before real use",
        "scope": "illustrative electricity factor",
    },
}


def get_factor(key: str = "grid_electricity_example") -> float:
    return float(EMISSION_FACTORS[key]["value"])
