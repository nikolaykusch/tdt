"""
cost_model.py
=============
Economic model: conversion of energy (J) and transmitted data volume
(bits) into monetary cost (USD), as well as the calculation of traffic savings from
Edge filtering.

This directly implements the indicators stated in the article's abstract/goals
("Bandwidth Cost", "traffic savings",
"energy consumption comparison"), which were absent in the initial version of the code.

All functions are pure (no side effects), which simplifies unit testing.
"""

from . import config


def joules_to_kwh(energy_j: float) -> float:
    return energy_j / 3_600_000.0


def bits_to_gb(bits: float) -> float:
    return bits / 8.0 / 1e9


def energy_cost_usd(energy_j: float) -> float:
    """Energy cost (J) at the assumed COST_PER_KWH_USD rate."""
    return joules_to_kwh(energy_j) * config.COST_PER_KWH_USD


def traffic_cost_usd(bits_transmitted: float) -> float:
    """Transmitted traffic cost (bits) at the assumed COST_PER_GB_USD rate."""
    return bits_to_gb(bits_transmitted) * config.COST_PER_GB_USD


def total_cost_usd(edge_energy_j: float, cloud_tx_energy_j: float, cloud_bits_transmitted: float) -> float:
    """
    Total cost of ownership (energy + traffic) for a single run.

    Edge node energy is accounted for at the electricity rate;
    Cloud traffic — at the telecom operator's rate; radio transmitter energy
    is added to the total energy component.
    """
    return (
        energy_cost_usd(edge_energy_j)
        + energy_cost_usd(cloud_tx_energy_j)
        + traffic_cost_usd(cloud_bits_transmitted)
    )


def traffic_savings_ratio(bits_attempted_hybrid: float, bits_raw_equivalent: float) -> float:
    """
    The fraction of traffic savings from Edge filtering:

        savings = 1 - (volume Edge decided to send in
                       HYBRID_FILTERED mode, BEFORE passing the radio channel)
                      / (volume that would have to be sent in
                         PARALLEL_RAW mode — "raw equivalent")

    IMPORTANT: the numerator must be taken as `bits_attempted` (the volume queued
    in the network segment BEFORE the radio channel), and NOT `bits_transmitted`
    (the volume that actually passed the radio channel). Using bits_transmitted
    mixes the effect of packet losses due to channel quality (e.g., in a tunnel)
    into the traffic savings, which has nothing to do with the architecture's
    decision on data filtering/compression.

    Returns a value in the range [0, 1]; 0 if there is nothing to compare with
    (bits_raw_equivalent == 0).
    """
    if bits_raw_equivalent <= 0:
        return 0.0
    return max(0.0, 1.0 - bits_attempted_hybrid / bits_raw_equivalent)