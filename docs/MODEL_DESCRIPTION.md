# Model Description: Code and Formulas Correspondence

This document explains the mathematical model implemented in the
`train_digital_twin` package and explicitly links each code fragment to a formula.
It is intended as a reference for reading the code and as a basis for the
"Methodology" section of the article.

## Table of Contents

1. [Discretization and general loop](#1-discretization-and-general-loop)
2. [Traffic](#2-traffic)
3. [Anomaly Burst](#3-anomaly-burst)
4. [Radio channel](#4-radio-channel)
5. [Edge node](#5-edge-node)
6. [Cloud subsystem](#6-cloud-subsystem)
7. [Economic model](#7-economic-model)
8. [System state classification](#8-system-state-classification)
9. [Architectural modes](#9-architectural-modes)
10. [Explicit model simplifications (applicability limits)](#10-explicit-model-simplifications-applicability-limits)

---

## 1. Discretization and general loop

The model is discrete-time (not discrete-event): the state is updated at
each tick `Δt = SIM_TICK_S = 0.01 s` using Euler's method
(`simulator.TrainDigitalTwin.step`). The sequence of actions per tick is:

1. Update distance/time based on the current segment speed.
2. Update Anomaly Burst (`traffic.AnomalyProcess.step`).
3. Generate traffic (`traffic.TrafficGenerator.generate`).
4. Edge processing (`edge_node.EdgeNode.step`).
5. Determine the volume going to the Cloud based on the architectural mode.
6. Channel calculation (`channel.RadioChannel.step`).
7. Cloud processing (`cloud_node.CloudNode.step`).
8. System state classification.
9. Write `models.SimMetrics`.

## 2. Traffic

Implemented in `traffic.TrafficGenerator.generate`.


```

λ_safety(t) = N_vib·f_s·B_vib·ξ_vib(t) + N_temp·f_temp·B_temp·ξ_temp(t)
λ_video(t)  = { R_min·ξ_v(t),  "low activity" state
{ R_max·ξ_v(t),  "high activity" state
λ_monitoring(t) = N_cam · λ_video(t) · [ANOMALY_VIDEO_MULTIPLIER, if anomaly is active]

```

MMPP state switching is a Bernoulli trial with probability
`MMPP_TRANSITION_PROB` at each tick. The temperature channel (`TEMP_*` in
`config.py`) is new compared to the initial version of the model, added to
match the article's abstract (vibration, temperature, and video sensors).

## 3. Anomaly Burst

Implemented in `traffic.AnomalyProcess`. A renewal process: in the inactive
state — a Bernoulli trial with probability `ANOMALY_PROB_PER_TICK`;
upon activation, it is held for `ANOMALY_DURATION_S` by a deterministic
timer.

Theoretical time share in the active state in steady mode:


```

P(active) ≈ (p_a · T_a) / (p_a · T_a + Δt)

```

This is used as an internal implementation correctness check (comparison
with the empirical share in the simulation results).

## 4. Radio channel

Implemented in `channel.RadioChannel.step`. Compared to the initial version,
the model has been supplemented with three elements (detailed in the file's docstring):

**4.1. Correlated shadow fading (AR(1) in the dB domain).**


```

τ_c ≈ 1 / (FADING_COHERENCE_FACTOR · f_d)
ρ   = exp(-Δt / τ_c)
SNR_dB(t) = μ + ρ·(SNR_dB(t-Δt) - μ) + σ·sqrt(1-ρ²)·N(0,1)

```

where `μ, σ` are `LOS_SNR_MEAN_DB/STD_DB` or `NLOS_SNR_MEAN_DB/STD_DB` depending
on the environment.

**4.2. Triangular droop near the cell edge.**


```

droop_dB = -(1 - triangular(pos_frac)) · PATHLOSS_CELL_EDGE_DROOP_DB

```

where `pos_frac ∈ [0,1]` is the relative position within the inter-handover interval.

**4.3. Doppler degradation (same as in the initial version).**


```

f_d = (v/c) · f_c
SNR_eff = 10^(SNR_dB/10) · exp(-f_d / DOPPLER_DECAY_HZ)

```

**4.4. Capacity with MCS derating.**


```

C_Shannon = B_w · log2(1 + SNR_eff)
C_realizable = C_Shannon · SPECTRAL_EFFICIENCY_FACTOR

```

**4.5. HARQ and residual BLER.**


```

BLER_single(SNR_dB_eff) = 1 / (1 + exp((SNR_dB_eff - BLER_THRESHOLD_DB) / BLER_STEEPNESS_DB))
BLER_after_HARQ = BLER_single ^ (HARQ_MAX_RETRANSMISSIONS + 1)
C_effective = C_realizable · (1 - BLER_after_HARQ)
extra_latency_HARQ ≈ HARQ_ROUND_TRIP_S · BLER_single

```

**4.6. Tunnel and handover (forced outage).**

Tunnel: `C(t) = 0` for the entire segment. Handover: `C(t) = 0` during
`HANDOVER_DURATION_S` — a **standalone parameter**, no longer tied
to `Δt`, as it was in the initial version of the model.

## 5. Edge node

Implemented in `edge_node.EdgeNode.step`.

**5.1. Packet queue with two priority classes (QoS).**

SAFETY (vibro+temperature) and MONITORING (video) data enter a shared
buffer `EDGE_BUFFER_MAX_BITS`, served with strict priority
(`queueing.PriorityQueueSystem`): SAFETY is served first with the full
available capacity; buffer overflow drops MONITORING first (tail-drop).

**5.2. Thermal dynamics and throttling — UNCHANGED from the initial version:**


```

μ(T_cpu) = { μ_max,                                    T_cpu ≤ T_crit
{ μ_max · exp(-α·(T_cpu - T_crit)) · ζ(t),  T_cpu > T_crit,  ζ(t)~U(0.9,1.0)

C_th · dT_cpu/dt = P_active·U(t) - (T_cpu - T_amb)/R_th

```

**5.3. Age of Information — separately per class (fix).**


```

AoI_safety(t)     = t - U_safety(t)
AoI_monitoring(t) = t - U_monitoring(t)
AoI_edge(t)       = max(AoI_safety(t), AoI_monitoring(t))

```

where `U_class(t)` is the generation time of the freshest fully processed
packet of that class. **Important:** a unified (non-class-based) AoI would be
misleading because the priority service of SAFETY masks the real
stagnation of the MONITORING queue (discovered by the
`test_edge_monitoring_aoi_is_nonzero_under_sustained_congestion` test).

**5.4. Energy.**


```

P(t) = EDGE_IDLE_POWER_W + POWER_ACTIVE_W · U(t)
E(t) = P(t) · Δt + [ML_INFERENCE_ENERGY_J_PER_MBIT · volume/1e6, if ML filtering is active]

```

Energy DOES NOT affect the thermal equation (5.2) — this is a separate accounting for
the economic model that does not alter the already verified thermal dynamics.

## 6. Cloud subsystem

Implemented in `cloud_node.CloudNode.step`, TWO sequential segments:

**6.1. Network segment** — a deadline queue (TTL reneging), just like in the
initial version, but now with QoS priority:


```

tx(t) = min(Q_network(t), C_effective(t)·Δt)
packet is dropped if (t - t_gen) > TTL_SECONDS

```

**6.2. Server segment (NEW)** — finite backend throughput
(multi-tenant load) + fixed processing delay:


```

L_server(t) = Q_server(t)/CLOUD_SERVER_CAPACITY_BPS + CLOUD_SERVER_FIXED_DELAY_S
L_cloud(t)  = L_network(t) + L_server(t)

```

AoI is calculated similarly to 5.3, separately per class, based on packets that
have passed BOTH segments.

## 7. Economic model

Implemented in `cost_model.py`. Previously did not exist in the model.


```

cost_traffic = (bits_sent / 8 / 1e9) · COST_PER_GB_USD
cost_energy  = (energy_J / 3.6e6) · COST_PER_KWH_USD
cost_total   = cost_traffic + cost_energy(edge) + cost_energy(radio_tx)

savings = 1 - (bits_attempted_HYBRID / bits_raw_equivalent_PARALLEL_RAW)

```

**Important nuance** (captured by the
`test_traffic_savings_uses_attempted_not_channel_affected_transmitted` test):
traffic savings are calculated from the volume QUEUED before the channel
(`bits_attempted`), NOT from the volume actually transmitted after the channel
(`bits_transmitted`), because the latter depends on channel quality
(tunnel losses, etc.) and does not reflect the architecture's filtering decision.
Traffic cost (`total_cost_usd`), on the other hand, is correctly calculated from
`bits_transmitted` — the telecom operator bills for the data actually transmitted.

## 8. System state classification

Unchanged from the initial version:


```

OK(L, PLR) ⟺ (L < LAT_SAFETY_THRESHOLD_S) ∧ (PLR < PLR_SAFETY_THRESHOLD)

Both OK / Cloud Down / Edge Down / Both Down — based on (Edge_OK, Cloud_OK)

```

## 9. Architectural modes

`models.ArchitectureMode` (detailed semantics description is in the class docstring):

- **PARALLEL_RAW** — legacy mode, reproduces the behavior of the initial model
  version one-to-one (Edge and Cloud receive the same full volume).
- **HYBRID_FILTERED** — realistic mode: SAFETY is always full, while
  MONITORING is compressed (`EDGE_TO_CLOUD_COMPRESSION_RATIO`), except for
  Anomaly Burst periods (then — full raw stream).

Both modes run in parallel for each factorial design combination
(`experiments/run_experiment.py`), allowing for a proper comparison of
"how much it would cost without Edge filtering" versus "how much it costs with it."

## 10. Explicit model simplifications (applicability limits)

- A single train in isolation (no inter-train interference).
- MAC layer is ignored, except for the aggregated HARQ/BLER effect.
- Single-node (lumped RC) thermal model.
- A "packet" is an aggregated volume per tick `Δt`, not an Ethernet/IP frame.
- All economic constants (`COST_PER_GB_USD`, `COST_PER_KWH_USD`,
  `RADIO_TX_POWER_W`, etc.) are documented ASSUMPTIONS of a plausible
  order of magnitude, not calibrated data from a specific operator/device.
- The Cloud server segment is considered reliable (lossless); real losses
  only occur in the network segment.

A detailed list with explanations is found in the `config.py` docstring next to each
parameter group (marked as `ASSUMPTION:`).

