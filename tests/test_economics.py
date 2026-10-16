from app.economics import annualize_period_energy, scenario


def test_scenario_calculates_expected_values():
    result = scenario(100000, 0.2, 0.8, 0.5)
    assert result["avoided_energy_kwh"] == 20000
    assert result["estimated_cost_saving"] == 16000
    assert result["avoided_co2_kg"] == 10000
    assert result["avoided_co2_t"] == 10



def test_annualize_period_energy_uses_inclusive_date_span():
    assert annualize_period_energy(100, observed_days=9) == 100 * 365 / 9


def test_annualize_period_energy_rejects_invalid_date_range():
    import pytest

    with pytest.raises(ValueError, match="At least one observed day"):
        annualize_period_energy(100, observed_days=0)
