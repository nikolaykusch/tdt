"""Integration tests for simulator.TrainDigitalTwin."""

import pytest

from train_digital_twin import TrainDigitalTwin, config
from train_digital_twin.models import ArchitectureMode


def _run(route_name, edge_profile, anomaly, mode, seed=42):
    route = config.ROUTES[route_name]
    twin = TrainDigitalTwin(
        route, run_id=0, scenario_name=route_name,
        edge_profile_name=edge_profile, anomaly_enabled=anomaly,
        architecture_mode=mode, seed=seed,
    )
    twin.run()
    return twin


def test_nominal_scenario_stays_within_safety_thresholds():
    twin = _run('Open terrain (350 km/h)', 'Nominal', False, ArchitectureMode.HYBRID_FILTERED)
    last = twin.metrics[-1]
    assert last.edge_plr == pytest.approx(0.0, abs=1e-9)
    assert last.system_state == 'Both OK'


def test_tunnel_scenario_degrades_cloud_but_not_edge():
    twin = _run('Deep tunnels (Connection loss)', 'Nominal', False, ArchitectureMode.HYBRID_FILTERED)
    last = twin.metrics[-1]
    assert last.edge_plr == pytest.approx(0.0, abs=1e-9)
    assert last.cloud_plr > 0.3   # significant losses due to tunnels
    cloud_down_share = sum(1 for m in twin.metrics if m.system_state == 'Cloud Down') / len(twin.metrics)
    assert cloud_down_share > 0.5


def test_overload_profile_triggers_thermal_throttling():
    twin = _run('Open terrain (350 km/h)', 'Overload', True, ArchitectureMode.PARALLEL_RAW)
    max_temp = max(m.edge_temp for m in twin.metrics)
    assert max_temp > config.T_CRIT


def test_degraded_profile_avoids_throttling_under_same_load():
    twin = _run('Mixed stress test', 'Degraded', True, ArchitectureMode.PARALLEL_RAW)
    max_temp = max(m.edge_temp for m in twin.metrics)
    assert max_temp < config.T_CRIT


def test_hybrid_mode_reduces_cloud_traffic_vs_parallel_raw():
    twin_raw = _run('Open terrain (350 km/h)', 'Nominal', False, ArchitectureMode.PARALLEL_RAW, seed=7)
    twin_hybrid = _run('Open terrain (350 km/h)', 'Nominal', False, ArchitectureMode.HYBRID_FILTERED, seed=7)

    assert twin_hybrid.total_cloud_bits_attempted < twin_raw.total_cloud_bits_attempted
    # in PARALLEL_RAW mode, the volume QUEUED (attempted) must
    # equal the "raw equivalent" (nothing is compressed); bits_sent
    # is intentionally NOT checked here, as it additionally depends on losses
    # in the radio channel (not an indication of the architectural filtering decision).
    assert twin_raw.total_cloud_bits_attempted == pytest.approx(
        twin_raw.total_cloud_bits_raw_equivalent, rel=0.01
    )


def test_edge_monitoring_aoi_is_nonzero_under_sustained_congestion():
    twin = _run('Open terrain (350 km/h)', 'Overload', True, ArchitectureMode.PARALLEL_RAW)
    mean_aoi = sum(m.edge_monitoring_aoi for m in twin.metrics) / len(twin.metrics)
    # MONITORING (video) has no priority and that is why it accumulates in the queue
    assert mean_aoi > 1.0
    # SAFETY, on the contrary, has service priority and must remain fresh
    mean_safety_aoi = sum(m.edge_safety_aoi for m in twin.metrics) / len(twin.metrics)
    assert mean_safety_aoi < mean_aoi


def test_edge_aoi_is_near_zero_under_nominal_load():
    twin = _run('Open terrain (350 km/h)', 'Nominal', False, ArchitectureMode.PARALLEL_RAW)
    mean_aoi = sum(m.edge_aoi for m in twin.metrics) / len(twin.metrics)
    # Short-term exceedances of the MMPP instantaneous rate (up to 24 Mbps at
    # mu_max=120 Mbps) can create millisecond queues even in
    # nominal mode — this is expected and is not an overload.
    assert mean_aoi < 0.05


def test_traffic_savings_uses_attempted_not_channel_affected_transmitted():
    """
    Regression test for a nuance discovered during development: in a bad channel
    (tunnel) bits_sent (after the channel) can be << bits_attempted (before the channel)
    even in PARALLEL_RAW mode, where no intentional filtering
    occurs. The traffic savings metric MUST be based on bits_attempted,
    otherwise "savings" falsely mixes in normal packet losses in the channel.
    """
    twin = _run('Deep tunnels (Connection loss)', 'Nominal', False, ArchitectureMode.PARALLEL_RAW)
    # In PARALLEL_RAW nothing is compressed -> attempted must exactly equal the raw equivalent
    assert twin.total_cloud_bits_attempted == pytest.approx(
        twin.total_cloud_bits_raw_equivalent, rel=0.01
    )
    # But actually transmitted (bits_sent) is significantly less due to losses in the tunnel
    assert twin.total_cloud_bits_sent < 0.7 * twin.total_cloud_bits_attempted


def test_cumulative_cost_is_monotonically_nondecreasing():
    twin = _run('Mixed stress test', 'Overload', True, ArchitectureMode.HYBRID_FILTERED)
    costs = [m.cumulative_cost_usd for m in twin.metrics]
    assert all(b >= a - 1e-9 for a, b in zip(costs, costs[1:]))