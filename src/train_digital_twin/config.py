"""
config.py
=========
Centralized configuration of the train digital twin simulator.

Each parameter is documented and, where it concerns empirical assumptions
(traffic cost, radio modem energy efficiency, etc.), marked as
ASSUMPTION with a brief justification of the order of magnitude. This allows
for easy re-evaluation of results when calibrated data from real telemetry becomes available.

File sections:
    1. Simulation discretization
    2. Traffic: vibro-diagnostics, temperature, video
    3. Anomaly Burst (event-driven load)
    4. Edge node: buffer, thermal model, energy, ML filtering
    5. Radio channel: Shannon, Doppler, fading, HARQ, handover
    6. Cloud: network and server segments
    7. Economic model (traffic and energy cost)
    8. Safety thresholds
    9. Routes
    10. Edge profiles
    11. Factorial experiment design
"""

from .models import RouteSegment, EnvType, ArchitectureMode

# ============================================================================
# 1. SIMULATION DISCRETIZATION
# ============================================================================

NUM_RUNS = 5
"""Number of independent Monte-Carlo replications per factor combination.
For final publication, NUM_RUNS >= 10 is recommended for narrower confidence
intervals (see experiments/run_experiment.py --num-runs)."""

SIM_TICK_S = 0.01
"""Discretization step Δt, s."""

# ============================================================================
# 2. TRAFFIC: VIBRO-DIAGNOSTICS, TEMPERATURE, VIDEO
# ============================================================================

# --- vibro-diagnostics (SAFETY class) ---
VIB_SENSORS = 8
VIB_FREQ_HZ = 50_000
VIB_DEPTH_BITS = 24
VIB_RATE_BPS = VIB_SENSORS * VIB_FREQ_HZ * VIB_DEPTH_BITS

# --- axle box temperature (SAFETY class; added to match the article's abstract) ---
TEMP_SENSORS = 8
"""Number of temperature sensors (one per axle box/bogie)."""
TEMP_FREQ_HZ = 10
"""Temperature sampling frequency, Hz. Thermal processes in the axle box are slow,
so 10 Hz more than covers the bearing heating dynamics."""
TEMP_DEPTH_BITS = 16
TEMP_RATE_BPS = TEMP_SENSORS * TEMP_FREQ_HZ * TEMP_DEPTH_BITS

# --- video stream (MONITORING class) ---
VIDEO_CAMERAS = 2
VIDEO_RATE_MIN_BPS = 4_000_000
VIDEO_RATE_MAX_BPS = 12_000_000
MMPP_TRANSITION_PROB = 0.05
"""Probability of switching between low/high MMPP states at each step."""

# ============================================================================
# 3. ANOMALY BURST (EVENT-DRIVEN LOAD)
# ============================================================================

ANOMALY_PROB_PER_TICK = 0.0008
ANOMALY_DURATION_S = 3.0
ANOMALY_VIDEO_MULTIPLIER = 5.0
ANOMALY_EXTRA_UTILIZATION = 0.4

# ============================================================================
# 4. EDGE NODE: BUFFER, THERMAL MODEL, ENERGY, ML FILTERING
# ============================================================================

EDGE_BUFFER_MAX_BITS = 300_000_000
"""Maximum Edge buffer capacity, bits. Shared between SAFETY/MONITORING queues
(SAFETY has priority when dropping excess — see queueing.PriorityQueueSystem)."""

# --- thermal model (baseline values; profiles in EDGE_PROFILES override them) ---
T_AMB = 25.0
T_CRIT = 75.0
CPU_MAX_RATE_BPS = 120_000_000
THERMAL_CAPACITY = 40.0
THERMAL_RESISTANCE = 0.6
POWER_ACTIVE_W = 35.0
ALPHA_THROTTLING = 0.15

# --- energy (for the economic model; DOES NOT affect the thermal equation,
#     so as not to change the already verified thermal dynamics) ---
EDGE_IDLE_POWER_W = 5.0
"""ASSUMPTION: baseline board power consumption in idle mode (without
computational load), W. Order of magnitude is typical for industrial
ARM/Jetson-like embedded modules."""

# --- ML filtering / hybrid architecture ---
EDGE_TO_CLOUD_COMPRESSION_RATIO = 0.05
"""Fraction of video traffic that Edge forwards to the Cloud in
ArchitectureMode.HYBRID_FILTERED mode under normal conditions (the rest is locally
processed features/metadata that do not require full video transmission).
ASSUMPTION: 5% corresponds to typical "raw video -> feature summary" compression
for anomaly detection tasks (e.g., periodic thumbnail frames + feature vectors
instead of full video stream)."""

ML_INFERENCE_EXTRA_UTILIZATION = 0.10
"""Additional CPU load on the Edge node from executing ML inference
filtering (fraction of mu_max), when ArchitectureMode.HYBRID_FILTERED."""

ML_INFERENCE_ENERGY_J_PER_MBIT = 0.02
"""ASSUMPTION: inference energy, J/Mbit of processed input data."""

# ============================================================================
# 5. RADIO CHANNEL: SHANNON, DOPPLER, FADING, HARQ, HANDOVER
# ============================================================================

FREQ_C_HZ = 2.1e9
C_SPEED = 3e8
BANDWIDTH_HZ = 20_000_000

# --- mean SNR and fading (dB domain, time-correlated AR(1)) ---
LOS_SNR_MEAN_DB = 29.5
LOS_SNR_STD_DB = 1.5
NLOS_SNR_MEAN_DB = 19.0
NLOS_SNR_STD_DB = 3.0
DOPPLER_DECAY_HZ = 500.0
"""Empirical decay constant of effective SNR due to Doppler shift:
SNR_eff = SNR * exp(-f_d / DOPPLER_DECAY_HZ)."""

PATHLOSS_CELL_EDGE_DROOP_DB = 2.0
"""Additional triangular SNR attenuation (dB) near the cell edge (before
handover) relative to the center of the inter-handover interval — a rough
approximation of the real received power profile along the route."""

FADING_COHERENCE_FACTOR = 4.0
"""Coefficient in the approximate channel coherence time formula
T_c ≈ 1 / (FADING_COHERENCE_FACTOR * f_d) (classical Clarke/Jakes estimation
up to order of magnitude), which determines the autocorrelation of the
Gauss-Markov AR(1) shadow fading process between ticks."""

# --- HARQ / residual BLER after retransmissions ---
HARQ_MAX_RETRANSMISSIONS = 2
HARQ_ROUND_TRIP_S = 0.008
"""Duration of one HARQ round (request-retry), s. Typical value for
LTE-like networks ~8 ms."""

BLER_THRESHOLD_DB = 5.0
"""Effective SNR value (dB) at which single-transmission BLER = 50%."""

BLER_STEEPNESS_DB = 3.0
"""Steepness of the sigmoid BLER(SNR) curve, dB."""

SPECTRAL_EFFICIENCY_FACTOR = 0.75
"""Fraction of theoretical Shannon capacity achievable by a real
modulation and coding scheme (MCS) considering protocol overhead."""

HANDOVER_DURATION_S = 0.05
"""Duration of forced connection drop during handover, s. IMPORTANT:
this is now an independent parameter, independent of the discretization step Δt
(unlike the previous version of the model, where the drop lasted exactly 1 tick)."""

TTL_SECONDS = 1.5

# ============================================================================
# 6. CLOUD: NETWORK AND SERVER SEGMENTS
# ============================================================================

CLOUD_SERVER_CAPACITY_BPS = 500_000_000
"""ASSUMPTION: effective backend throughput (processing, DB write)
per train under multi-tenant load from parallel trains, bps."""

CLOUD_SERVER_FIXED_DELAY_S = 0.02
"""Fixed server processing delay (message queue write, DB, etc.), s."""

RADIO_TX_POWER_W = 2.0
"""ASSUMPTION: onboard radio modem transmitter power in active transmission
mode, W (typical value for an industrial LTE/5G modem)."""

# ============================================================================
# 7. ECONOMIC MODEL (TRAFFIC AND ENERGY COST)
# ============================================================================

COST_PER_GB_USD = 5.0
"""ASSUMPTION: cost of transmitting 1 GB via industrial
LTE-R/satellite channel with roaming, USD. Value is easily adjusted
for a specific operator contract."""

COST_PER_KWH_USD = 0.15
"""ASSUMPTION: average electricity cost, USD/kWh."""

# ============================================================================
# 8. SAFETY THRESHOLDS
# ============================================================================

LAT_SAFETY_THRESHOLD_S = 0.15
PLR_SAFETY_THRESHOLD = 0.02

# ============================================================================
# 9. ROUTES
# ============================================================================

ROUTES = {
    'Open terrain (350 km/h)': [
        RouteSegment(15000, 350, EnvType.LOS, 3000),
        RouteSegment(15000, 350, EnvType.LOS, 3000),
        RouteSegment(15000, 350, EnvType.LOS, 3000),
    ],
    'Deep tunnels (Connection loss)': [
        RouteSegment(5000, 250, EnvType.LOS, 2000),
        RouteSegment(8000, 250, EnvType.TUNNEL, 8000),
        RouteSegment(4000, 250, EnvType.LOS, 2000),
        RouteSegment(6000, 250, EnvType.TUNNEL, 6000),
        RouteSegment(5000, 250, EnvType.LOS, 2000),
    ],
    'Urban area (NLOS)': [
        RouteSegment(4000, 120, EnvType.NLOS, 800),
        RouteSegment(4000, 160, EnvType.NLOS, 800),
        RouteSegment(4000, 100, EnvType.NLOS, 500),
        RouteSegment(4000, 140, EnvType.NLOS, 800),
    ],
    'Mixed stress test': [
        RouteSegment(6000, 300, EnvType.LOS, 2500),
        RouteSegment(3000, 280, EnvType.TUNNEL, 3000),
        RouteSegment(5000, 180, EnvType.NLOS, 1000),
        RouteSegment(4000, 300, EnvType.LOS, 2500),
        RouteSegment(2000, 200, EnvType.TUNNEL, 2000),
    ],
}

# ============================================================================
# 10. EDGE PROFILES
# ============================================================================

EDGE_PROFILES = {
    'Nominal': dict(
        T_AMB=25.0,
        THERMAL_RESISTANCE=0.6,
        CPU_MAX_RATE_BPS=120_000_000,
        ALPHA_THROTTLING=0.15,
    ),
    'Degraded': dict(
        T_AMB=45.0,
        THERMAL_RESISTANCE=1.4,
        CPU_MAX_RATE_BPS=90_000_000,
        ALPHA_THROTTLING=0.30,
    ),
    'Overload': dict(
        T_AMB=50.0,
        THERMAL_RESISTANCE=1.8,
        CPU_MAX_RATE_BPS=60_000_000,
        ALPHA_THROTTLING=0.45,
    ),
}

# ============================================================================
# 11. FACTORIAL EXPERIMENT DESIGN
# ============================================================================
# Each combination is executed in two architecture modes (PARALLEL_RAW as a
# baseline for compatibility with the previous model version, HYBRID_FILTERED as
# the main realistic mode) — see experiments/run_experiment.py.

EXPERIMENT_MATRIX = [
    dict(route='Open terrain (350 km/h)', edge_profile='Nominal', anomaly=False),
    dict(route='Deep tunnels (Connection loss)', edge_profile='Nominal', anomaly=False),
    dict(route='Urban area (NLOS)', edge_profile='Nominal', anomaly=False),
    dict(route='Open terrain (350 km/h)', edge_profile='Overload', anomaly=True),
    dict(route='Mixed stress test', edge_profile='Overload', anomaly=True),
    dict(route='Mixed stress test', edge_profile='Degraded', anomaly=True),
]

DEFAULT_ARCHITECTURE_MODE = ArchitectureMode.HYBRID_FILTERED