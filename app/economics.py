"""Transparent scenario economics for building-energy screening.

The calculator estimates avoided energy and simplified energy-charge savings.
It is not a full utility-bill model: demand charges, time-of-use tariffs,
taxes, contract terms and rebound effects are outside this prototype.
"""


def scenario(
    annual_energy_kwh: float,
    reduction_pct: float,
    tariff_per_kwh: float,
    emission_factor_kg_per_kwh: float,
):
    annual_energy_kwh = max(float(annual_energy_kwh), 0.0)
    reduction_pct = min(max(float(reduction_pct), 0.0), 1.0)
    tariff_per_kwh = max(float(tariff_per_kwh), 0.0)
    emission_factor_kg_per_kwh = max(float(emission_factor_kg_per_kwh), 0.0)

    avoided_energy_kwh = annual_energy_kwh * reduction_pct
    energy_charge_saving = avoided_energy_kwh * tariff_per_kwh
    avoided_co2_kg = avoided_energy_kwh * emission_factor_kg_per_kwh

    return {
        "avoided_energy_kwh": avoided_energy_kwh,
        "estimated_energy_charge_saving": energy_charge_saving,
        "estimated_cost_saving": energy_charge_saving,
        "avoided_co2e_kg": avoided_co2_kg,
        "avoided_co2e_t": avoided_co2_kg / 1000.0,
    }
