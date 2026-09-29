"""Transparent scenario economics for the building low-carbon demo.

All outputs are estimates. Real tariffs, demand charges and contracts must be
verified before business decisions are made.
"""


def scenario(annual_energy_kwh: float, reduction_pct: float, tariff_per_kwh: float, emission_factor_kg_per_kwh: float):
    annual_energy_kwh = max(float(annual_energy_kwh), 0.0)
    reduction_pct = min(max(float(reduction_pct), 0.0), 1.0)
    tariff_per_kwh = max(float(tariff_per_kwh), 0.0)
    emission_factor_kg_per_kwh = max(float(emission_factor_kg_per_kwh), 0.0)

    avoided_energy_kwh = annual_energy_kwh * reduction_pct
    cost_saving = avoided_energy_kwh * tariff_per_kwh
    avoided_co2e_kg = avoided_energy_kwh * emission_factor_kg_per_kwh

    return {
        "avoided_energy_kwh": avoided_energy_kwh,
        "estimated_cost_saving": cost_saving,
        "avoided_co2e_kg": avoided_co2e_kg,
        "avoided_co2e_t": avoided_co2e_kg / 1000.0,
    }
