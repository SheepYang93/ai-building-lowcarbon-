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
        # Configured factors are kgCO2/kWh, not full life-cycle CO2e factors.
        "avoided_co2_kg": avoided_co2_kg,
        "avoided_co2_t": avoided_co2_kg / 1000.0,
    }


def annualize_period_energy(total_energy: float, observed_days: int, days_per_year: float = 365.0) -> float:
    """Annualize a total using average energy per observed day.

    Missing calendar dates are not treated as zero-consumption days. This is
    still a simple screening extrapolation, not a seasonal or weather-adjusted
    forecast; sparse or unrepresentative observations can bias the estimate.
    """
    total_energy = max(float(total_energy), 0.0)
    observed_days = int(observed_days)
    if observed_days < 1:
        raise ValueError("At least one observed day is required.")
    return total_energy * float(days_per_year) / observed_days
