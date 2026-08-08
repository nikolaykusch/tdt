"""Tests for channel.RadioChannel."""

import random

import pytest

from train_digital_twin import config
from train_digital_twin.channel import RadioChannel
from train_digital_twin.models import EnvType, RouteSegment


def test_tunnel_forces_zero_capacity():
    ch = RadioChannel(random.Random(1))
    seg = RouteSegment(length_m=1000, speed_kmh=200, env_type=EnvType.TUNNEL, handover_interval_m=500)
    sample = ch.step(seg, distance=100.0, dt=0.01)
    assert sample.capacity_bps == 0.0
    assert sample.bler == 1.0


def test_handover_forces_outage_for_configured_duration():
    ch = RadioChannel(random.Random(2))
    seg = RouteSegment(length_m=10000, speed_kmh=200, env_type=EnvType.LOS, handover_interval_m=100)

    # move until we cross the handover threshold
    dt = 0.01
    distance = 0.0
    v = 200 / 3.6
    outage_ticks = 0
    total_ticks = 0
    triggered = False
    for _ in range(500):
        distance += v * dt
        sample = ch.step(seg, distance, dt)
        total_ticks += 1
        if sample.capacity_bps == 0.0:
            outage_ticks += 1
            triggered = True
        elif triggered:
            break

    assert triggered
    expected_outage_ticks = round(config.HANDOVER_DURATION_S / dt)
    # +/-1 tick of rounding error is acceptable
    assert abs(outage_ticks - expected_outage_ticks) <= 1


def test_los_capacity_far_exceeds_nlos_on_average():
    rng = random.Random(3)
    ch_los = RadioChannel(rng)
    ch_nlos = RadioChannel(rng)
    seg_los = RouteSegment(length_m=100000, speed_kmh=100, env_type=EnvType.LOS, handover_interval_m=100000)
    seg_nlos = RouteSegment(length_m=100000, speed_kmh=100, env_type=EnvType.NLOS, handover_interval_m=100000)

    los_caps = [ch_los.step(seg_los, 10.0 + i, 0.01).capacity_bps for i in range(200)]
    nlos_caps = [ch_nlos.step(seg_nlos, 10.0 + i, 0.01).capacity_bps for i in range(200)]

    assert sum(los_caps) / len(los_caps) > sum(nlos_caps) / len(nlos_caps)


def test_bler_decreases_with_increasing_snr():
    high_bler = RadioChannel._bler(config.BLER_THRESHOLD_DB - 10)
    low_bler = RadioChannel._bler(config.BLER_THRESHOLD_DB + 10)
    assert high_bler > 0.9
    assert low_bler < 0.1
    assert high_bler > low_bler