"""
models.py
=========
Domain data models of the train digital twin: enumerated types and
data structures shared across all modules of the package.

Contains no computational logic — only structure definitions.
"""

from dataclasses import dataclass
from enum import Enum


class EnvType(Enum):
    """
    Radio channel environment type along the route.

    LOS:
        Line of Sight — direct visibility, best communication conditions.
    NLOS:
        Non-Line of Sight — absence of direct visibility (dense urban area).
    TUNNEL:
        Tunnel / total radio shadow — forced outage (C(t) = 0).
    """
    LOS = "LOS"
    NLOS = "NLOS"
    TUNNEL = "TUNNEL"


class ArchitectureMode(Enum):
    """
    Computing architecture mode — determines what volume of data the Edge node
    actually forwards to the Cloud subsystem.

    PARALLEL_RAW:
        Legacy mode, compatible with the initial version of the model: Edge and Cloud
        receive EXACTLY the same full volume of raw data independently of each
        other. Used as a baseline for comparison (without the effect of
        local filtering), and also allows reproducing the results of the
        previous simulator version one-to-one.

    HYBRID_FILTERED:
        Realistic mode modeling the actual value of Edge computing: Edge
        always receives and analyzes the full stream (this is necessary for local
        real-time fault detection), however, only a compressed/aggregated version
        of the video stream is forwarded to the Cloud (a fraction
        `EDGE_TO_CLOUD_COMPRESSION_RATIO` of the original video traffic volume).
        Vibro-diagnostics and temperature data are forwarded to the Cloud fully
        (their volume is already small). During periods of active anomaly, Edge additionally
        forwards the full raw video stream to the Cloud for dispatcher
        verification (typical "summary normally, raw evidence on alarm" practice).
    """
    PARALLEL_RAW = "parallel_raw"
    HYBRID_FILTERED = "hybrid_filtered"


class PriorityClass(Enum):
    """
    Traffic priority class in service queues (QoS).

    SAFETY:
        Vibro-diagnostics + axle box temperature. Safety-critical traffic:
        served in queues first (strict priority), and upon
        buffer overflow is dropped last.
    MONITORING:
        Video stream. Best-effort monitoring traffic: served by whatever
        is left of the throughput after SAFETY, and is dropped
        first upon buffer overflow.
    """
    SAFETY = "safety"
    MONITORING = "monitoring"


@dataclass(frozen=True)
class RouteSegment:
    """
    A single homogeneous route segment.

    Attributes:
        length_m: segment length, m.
        speed_kmh: train speed on the segment, km/h.
        env_type: radio channel environment type (EnvType).
        handover_interval_m: distance between forced handovers, m.
    """
    length_m: float
    speed_kmh: float
    env_type: EnvType
    handover_interval_m: float


@dataclass
class SimMetrics:
    """
    A complete snapshot of the system state at a single simulation step.

    Metrics are grouped by purpose:
    - scenario identification and time/space coordinates;
    - Edge subsystem state (queue, temperature, QoS metrics, energy);
    - Cloud subsystem state (channel, queue, server, energy/cost);
    - anomaly mode and classified system state;
    - derived economic indicators (traffic/cost/energy).
    """
    # --- identification ---
    run_id: int
    scenario: str
    edge_profile: str
    architecture_mode: str
    anomaly_on: bool
    timestamp: float
    distance: float
    speed: float
    env_type: str

    # --- Edge subsystem ---
    edge_queue_bits: float
    edge_temp: float
    edge_latency: float
    edge_plr: float
    edge_aoi: float
    edge_safety_aoi: float
    edge_monitoring_aoi: float
    edge_safety_plr: float
    edge_monitoring_plr: float
    edge_energy_j: float

    # --- Cloud subsystem (network segment) ---
    cloud_queue_bits: float
    cloud_capacity: float
    cloud_harq_bler: float
    cloud_latency: float
    cloud_plr: float
    cloud_aoi: float
    cloud_safety_aoi: float
    cloud_monitoring_aoi: float
    cloud_safety_plr: float
    cloud_monitoring_plr: float

    # --- Cloud subsystem (server segment) ---
    cloud_server_latency: float

    # --- economics/energy ---
    cloud_bits_sent: float
    cloud_bits_attempted: float
    cloud_bits_raw_equivalent: float
    edge_energy_cumulative_j: float
    cloud_tx_energy_cumulative_j: float
    cumulative_cost_usd: float

    # --- anomaly and classification ---
    anomaly_active: bool
    system_state: str