"""Tests for cost_model.py."""

import pytest

from train_digital_twin import config, cost_model


def test_traffic_cost_matches_configured_rate():
    one_gb_in_bits = 8 * 1e9
    cost = cost_model.traffic_cost_usd(one_gb_in_bits)
    assert cost == pytest.approx(config.COST_PER_GB_USD)


def test_energy_cost_matches_configured_rate():
    one_kwh_in_joules = 3_600_000.0
    cost = cost_model.energy_cost_usd(one_kwh_in_joules)
    assert cost == pytest.approx(config.COST_PER_KWH_USD)


def test_traffic_savings_ratio_zero_when_no_reduction():
    assert cost_model.traffic_savings_ratio(100.0, 100.0) == pytest.approx(0.0)


def test_traffic_savings_ratio_full_when_no_bits_sent():
    assert cost_model.traffic_savings_ratio(0.0, 100.0) == pytest.approx(1.0)


def test_traffic_savings_ratio_typical_case():
    # sent 5, as the equivalent of raw data - 100 => 95% savings
    assert cost_model.traffic_savings_ratio(5.0, 100.0) == pytest.approx(0.95)


def test_traffic_savings_ratio_handles_zero_baseline():
    assert cost_model.traffic_savings_ratio(5.0, 0.0) == pytest.approx(0.0)