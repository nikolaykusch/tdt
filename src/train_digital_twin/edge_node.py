"""
edge_node.py
============
Edge node (onboard computer) model: packet queue with QoS priorities,
thermal dynamics with throttling, energy consumption accounting.

The thermal equation (7)-(9) is left UNCHANGED relative to the verified
previous version of the model — none of the new features (ML filtering,
QoS prioritization) affects the temperature other than through utilization
u(t), which was already the only input to the thermal model.
"""

import math
import random
from dataclasses import dataclass

from . import config
from .queueing import PriorityQueueSystem


@dataclass
class EdgeStepResult:
    latency: float
    plr: float
    safety_plr: float
    monitoring_plr: float
    aoi: float
    safety_aoi: float
    monitoring_aoi: float
    temperature: float
    queue_bits: float
    energy_j: float                 # energy consumed EXACTLY at this step
    energy_cumulative_j: float


class EdgeNode:
    """Edge node state for a single run."""

    def __init__(self, edge_profile_name: str, ml_filtering_enabled: bool,
                 rng: random.Random | None = None):
        self._rng = rng or random.Random()
        profile = config.EDGE_PROFILES[edge_profile_name]

        self.t_amb = profile['T_AMB']
        self.thermal_resistance = profile['THERMAL_RESISTANCE']
        self.cpu_max_rate_bps = profile['CPU_MAX_RATE_BPS']
        self.alpha_throttling = profile['ALPHA_THROTTLING']

        self.ml_filtering_enabled = ml_filtering_enabled

        self.t_cpu = self.t_amb
        self.queues = PriorityQueueSystem(config.EDGE_BUFFER_MAX_BITS)

        self.total_generated = 0.0
        self.total_safety_generated = 0.0
        self.total_monitoring_generated = 0.0
        self.total_dropped = 0.0
        self.total_safety_dropped = 0.0
        self.total_monitoring_dropped = 0.0

        self.last_delivered_gen_time_safety = 0.0
        self.last_delivered_gen_time_monitoring = 0.0
        self.energy_cumulative_j = 0.0

    def step(self, t: float, dt: float, safety_bits: float, monitoring_bits: float,
             anomaly_active: bool) -> EdgeStepResult:
        # --- arrival of new data ---
        self.queues.enqueue(t, safety_bits, monitoring_bits)
        self.total_generated += safety_bits + monitoring_bits
        self.total_safety_generated += safety_bits
        self.total_monitoring_generated += monitoring_bits

        # --- processing rate considering thermal throttling (9) ---
        mu_max = self.cpu_max_rate_bps
        if self.t_cpu > config.T_CRIT:
            mu = mu_max * math.exp(-self.alpha_throttling * (self.t_cpu - config.T_CRIT))
            mu *= self._rng.uniform(0.9, 1.0)
        else:
            mu = mu_max

        processed = min(self.queues.total_bits, mu * dt)
        serve_result = self.queues.serve(processed)

        # --- buffer overflow: priority dropping of MONITORING ---
        safety_dropped, monitoring_dropped = self.queues.enforce_buffer()
        self.total_dropped += safety_dropped + monitoring_dropped
        self.total_safety_dropped += safety_dropped
        self.total_monitoring_dropped += monitoring_dropped

        # --- utilization and thermal dynamics (7)-(8), UNCHANGED ---
        u_t = processed / (mu_max * dt) if mu_max > 0 else 0.0
        if anomaly_active:
            u_t = min(1.0, u_t + config.ANOMALY_EXTRA_UTILIZATION)
        if self.ml_filtering_enabled:
            u_t = min(1.0, u_t + config.ML_INFERENCE_EXTRA_UTILIZATION)

        dT = dt * (
            (config.POWER_ACTIVE_W * u_t) -
            (self.t_cpu - self.t_amb) / self.thermal_resistance
        ) / config.THERMAL_CAPACITY
        self.t_cpu += dT

        # --- quality metrics ---
        latency = self.queues.total_bits / mu if mu > 0 else float('inf')
        plr = self.total_dropped / self.total_generated if self.total_generated > 0 else 0.0
        safety_plr = (self.total_safety_dropped / self.total_safety_generated
                      if self.total_safety_generated > 0 else 0.0)
        monitoring_plr = (self.total_monitoring_dropped / self.total_monitoring_generated
                           if self.total_monitoring_generated > 0 else 0.0)

        safety_delivered_times, monitoring_delivered_times = serve_result['safety'][0], serve_result['monitoring'][0]
        if safety_delivered_times:
            self.last_delivered_gen_time_safety = max(safety_delivered_times)
        if monitoring_delivered_times:
            self.last_delivered_gen_time_monitoring = max(monitoring_delivered_times)

        safety_aoi = t - self.last_delivered_gen_time_safety
        monitoring_aoi = t - self.last_delivered_gen_time_monitoring
        # System AoI = age of the OLDEST not yet updated data category:
        # the system as a whole is "no fresher" than its worst served traffic class.
        # Combining classes into a single AoI (as it was in the previous
        # step of the model development) falsely masked the stagnation of the MONITORING queue
        # by instantaneous deliveries of priority SAFETY traffic.
        aoi = max(safety_aoi, monitoring_aoi)

        # --- energy: baseline consumption + active work + ML inference ---
        power_w = config.EDGE_IDLE_POWER_W + config.POWER_ACTIVE_W * u_t
        energy_j = power_w * dt
        if self.ml_filtering_enabled:
            energy_j += config.ML_INFERENCE_ENERGY_J_PER_MBIT * ((safety_bits + monitoring_bits) / 1e6)
        self.energy_cumulative_j += energy_j

        return EdgeStepResult(
            latency=latency,
            plr=plr,
            safety_plr=safety_plr,
            monitoring_plr=monitoring_plr,
            aoi=aoi,
            safety_aoi=safety_aoi,
            monitoring_aoi=monitoring_aoi,
            temperature=self.t_cpu,
            queue_bits=self.queues.total_bits,
            energy_j=energy_j,
            energy_cumulative_j=self.energy_cumulative_j,
        )