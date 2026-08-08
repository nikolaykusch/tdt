"""
traffic.py
==========
Generation of the digital twin's input information stream.

Traffic is divided into two QoS classes (PriorityClass), which is consistent with
packet accounting in queueing.PriorityQueueSystem:

    SAFETY:
        Vibro-diagnostics (deterministic bitrate ± multiplicative noise)
        + axle box temperature (new channel, added to match
        the article's abstract — previously missing in the model).

    MONITORING:
        Video stream with a two-state MMPP model (VIDEO_RATE_MIN/MAX_BPS),
        which is additionally amplified by ANOMALY_VIDEO_MULTIPLIER times during
        an active Anomaly Burst.

Formulas correspond to equations (1)-(2) of the article's methodological section;
see docs/MODEL_DESCRIPTION.md.
"""

import random
from dataclasses import dataclass

from . import config


@dataclass
class TrafficSample:
    """Data volume (in bits) generated in one step Δt, by classes."""
    safety_bits: float
    monitoring_bits: float

    @property
    def total_bits(self) -> float:
        return self.safety_bits + self.monitoring_bits


class TrafficGenerator:
    """
    Traffic generator state for a single run (stores the current MMPP state
    between steps).
    """

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self.video_state_high = False

    def generate(self, dt: float, anomaly_active: bool) -> TrafficSample:
        """Generates SAFETY and MONITORING traffic volume per step dt."""
        # --- SAFETY: vibro-diagnostics + temperature ---
        vib_bits = config.VIB_RATE_BPS * self._rng.uniform(0.95, 1.05) * dt
        temp_bits = config.TEMP_RATE_BPS * self._rng.uniform(0.95, 1.05) * dt
        safety_bits = vib_bits + temp_bits

        # --- MONITORING: video with MMPP + Anomaly Burst ---
        if self._rng.random() < config.MMPP_TRANSITION_PROB:
            self.video_state_high = not self.video_state_high

        vid_rate = config.VIDEO_RATE_MAX_BPS if self.video_state_high else config.VIDEO_RATE_MIN_BPS
        vid_rate *= self._rng.uniform(0.9, 1.1)

        if anomaly_active:
            vid_rate *= config.ANOMALY_VIDEO_MULTIPLIER

        monitoring_bits = config.VIDEO_CAMERAS * vid_rate * dt

        return TrafficSample(safety_bits=safety_bits, monitoring_bits=monitoring_bits)


class AnomalyProcess:
    """
    Binary process of event-driven Anomaly Burst load (equation (3)).

    Implemented as a renewal process: in the inactive
    state at each step — a Bernoulli trial with probability
    ANOMALY_PROB_PER_TICK; upon activation, the state is held for a fixed
    duration ANOMALY_DURATION_S.
    """

    def __init__(self, enabled: bool, rng: random.Random | None = None):
        self.enabled = enabled
        self._rng = rng or random.Random()
        self.active = False
        self._timer = 0.0

    def step(self, dt: float) -> bool:
        """Updates the state by one step and returns the new active state."""
        if not self.enabled:
            return False

        if self.active:
            self._timer -= dt
            if self._timer <= 0:
                self.active = False
        else:
            if self._rng.random() < config.ANOMALY_PROB_PER_TICK:
                self.active = True
                self._timer = config.ANOMALY_DURATION_S

        return self.active