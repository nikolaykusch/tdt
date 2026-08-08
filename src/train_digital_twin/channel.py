"""
channel.py
==========
HSR wireless communication channel model.

Compared to the initial version, the model has been supplemented with three realism elements
(detailed justification — docs/MODEL_DESCRIPTION.md, Sec. 3):

1. Correlated shadow fading. Previously, SNR was drawn independently at
   each step (Δt=10 ms), which neglects channel coherence time. Now
   SNR (in dB domain) is modeled by a Gauss-Markov AR(1) process with
   autocorrelation depending on the instantaneous Doppler shift f_d:
   T_c ≈ 1 / (FADING_COHERENCE_FACTOR * f_d).

2. Triangular SNR attenuation profile near the cell edge (before
   handover) — an approximation of the real received power profile
   along the route, instead of a constant statistical range.

3. HARQ / residual BLER. Instead of a rigid "all or nothing" at the
   Shannon capacity level, a probabilistic block error (BLER) model is added
   as a sigmoid function of effective SNR, combining
   HARQ (up to HARQ_MAX_RETRANSMISSIONS retries), which lowers residual BLER
   while simultaneously adding expected retransmission delay.

4. Handover now has a standalone duration HANDOVER_DURATION_S, not
   exactly one discretization step Δt, as before.
"""

import math
import random
from dataclasses import dataclass

from . import config
from .models import EnvType, RouteSegment


@dataclass
class ChannelSample:
    """Channel calculation result per step."""
    capacity_bps: float          # effective throughput after HARQ/MCS derating
    harq_extra_latency_s: float  # expected extra delay due to HARQ retries
    bler: float                  # instantaneous (single-shot) block error probability


class RadioChannel:
    """
    Radio channel state for a single run (maintains AR(1) fading state and
    remaining forced handover outage time between steps).
    """

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self._shadow_db_los = config.LOS_SNR_MEAN_DB
        self._shadow_db_nlos = config.NLOS_SNR_MEAN_DB
        self.last_handover_dist = 0.0
        self._handover_outage_remaining_s = 0.0

    def _update_shadowing(self, env: EnvType, f_d: float, dt: float) -> float:
        """Updates the shadow fading AR(1) process and returns the current
        SNR value in dB for a given environment (LOS/NLOS)."""
        if env == EnvType.LOS:
            mean_db, std_db = config.LOS_SNR_MEAN_DB, config.LOS_SNR_STD_DB
            prev = self._shadow_db_los
        else:
            mean_db, std_db = config.NLOS_SNR_MEAN_DB, config.NLOS_SNR_STD_DB
            prev = self._shadow_db_nlos

        tau_coherence = 1.0 / (config.FADING_COHERENCE_FACTOR * max(f_d, 1e-6))
        rho = math.exp(-dt / tau_coherence)

        new_val = mean_db + rho * (prev - mean_db) + std_db * math.sqrt(max(1 - rho ** 2, 0.0)) * self._rng.gauss(0, 1)

        if env == EnvType.LOS:
            self._shadow_db_los = new_val
        else:
            self._shadow_db_nlos = new_val
        return new_val

    @staticmethod
    def _cell_edge_droop_db(distance: float, last_handover_dist: float, handover_interval_m: float) -> float:
        """Triangular SNR attenuation (dB) depending on the position in the inter-handover
        interval: 0 in the center, -PATHLOSS_CELL_EDGE_DROOP_DB at the edges."""
        if handover_interval_m <= 0:
            return 0.0
        pos_frac = (distance - last_handover_dist) / handover_interval_m
        pos_frac = min(max(pos_frac, 0.0), 1.0)
        triangular = 1.0 - abs(2 * pos_frac - 1.0)  # 0 at edges, 1 in center
        return -(1.0 - triangular) * config.PATHLOSS_CELL_EDGE_DROOP_DB

    @staticmethod
    def _bler(snr_eff_db: float) -> float:
        """Sigmoid model of single-transmission residual BLER vs SNR (dB)."""
        x = (snr_eff_db - config.BLER_THRESHOLD_DB) / config.BLER_STEEPNESS_DB
        return 1.0 / (1.0 + math.exp(x))

    def step(self, segment: RouteSegment, distance: float, dt: float) -> ChannelSample:
        """Calculates effective channel capacity at step dt."""

        # --- forced outage: tunnel ---
        if segment.env_type == EnvType.TUNNEL:
            return ChannelSample(capacity_bps=0.0, harq_extra_latency_s=0.0, bler=1.0)

        # --- forced outage: handover (now with its own duration) ---
        if self._handover_outage_remaining_s > 0:
            self._handover_outage_remaining_s -= dt
            return ChannelSample(capacity_bps=0.0, harq_extra_latency_s=0.0, bler=1.0)

        if distance - self.last_handover_dist >= segment.handover_interval_m:
            self.last_handover_dist = distance
            self._handover_outage_remaining_s = config.HANDOVER_DURATION_S
            return ChannelSample(capacity_bps=0.0, harq_extra_latency_s=0.0, bler=1.0)

        # --- Doppler shift (4) ---
        v_ms = segment.speed_kmh / 3.6
        f_d = (v_ms / config.C_SPEED) * config.FREQ_C_HZ

        # --- correlated shadow fading + triangular droop ---
        shadow_db = self._update_shadowing(segment.env_type, f_d, dt)
        droop_db = self._cell_edge_droop_db(distance, self.last_handover_dist, segment.handover_interval_m)
        snr_db = shadow_db + droop_db

        snr_linear = 10 ** (snr_db / 10.0)
        snr_eff = snr_linear * math.exp(-f_d / config.DOPPLER_DECAY_HZ)
        snr_eff_db = 10 * math.log10(max(snr_eff, 1e-12))

        # --- Shannon capacity with MCS/protocol derating (5) ---
        shannon_bps = config.BANDWIDTH_HZ * math.log2(1 + snr_eff)
        realizable_bps = shannon_bps * config.SPECTRAL_EFFICIENCY_FACTOR

        # --- HARQ: residual BLER after up to HARQ_MAX_RETRANSMISSIONS retries ---
        bler_single = self._bler(snr_eff_db)
        bler_after_harq = bler_single ** (config.HARQ_MAX_RETRANSMISSIONS + 1)
        effective_bps = realizable_bps * (1.0 - bler_after_harq)

        # expected extra delay due to retries (linear approximation
        # for low/moderate BLER: E[extra rounds] ≈ bler_single)
        harq_extra_latency = config.HARQ_ROUND_TRIP_S * bler_single

        return ChannelSample(
            capacity_bps=effective_bps,
            harq_extra_latency_s=harq_extra_latency,
            bler=bler_single,
        )