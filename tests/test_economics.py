from app.economics import scenario


def test_scenario_calculates_expected_values():
    result = scenario(100000, 0.2, 0.8, 0.5)
    assert result["avoided_energy_kwh"] == 20000
    assert result["estimated_cost_saving"] == 16000
    assert result["avoided_co2e_kg"] == 10000
    assert result["avoided_co2e_t"] == 10
