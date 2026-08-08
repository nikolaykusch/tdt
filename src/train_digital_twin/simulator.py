"""
simulator.py
============
Main orchestrator of the train digital twin: the TrainDigitalTwin class
connects the traffic generator, radio channel, Edge and Cloud nodes into a single
discrete-time simulation loop of one run.

The architectural mode (ArchitectureMode) determines what volume of data the Edge
actually forwards to the Cloud at each step — see models.ArchitectureMode
for a detailed description of the semantics of PARALLEL_RAW versus HYBRID_FILTERED.
"""

import random

from . import config
from . import cost_model
from .channel import RadioChannel
from .cloud_node import CloudNode
from .edge_node import EdgeNode
from .models import ArchitectureMode, SimMetrics
from .traffic import AnomalyProcess, TrafficGenerator


class TrainDigitalTwin:
    """Train digital twin for a single run of one scenario."""

    def __init__(self, route, run_id, scenario_name='default',
                 edge_profile_name='Nominal', anomaly_enabled=True,
                 architecture_mode: ArchitectureMode = config.DEFAULT_ARCHITECTURE_MODE,
                 seed=None):
        self.route = route
        self.run_id = run_id
        self.scenario_name = scenario_name
        self.edge_profile_name = edge_profile_name
        self.architecture_mode = architecture_mode

        self._rng = random.Random(seed)

        self.traffic_gen = TrafficGenerator(self._rng)
        self.anomaly = AnomalyProcess(anomaly_enabled, self._rng)
        self.channel = RadioChannel(self._rng)

        ml_filtering_enabled = (architecture_mode == ArchitectureMode.HYBRID_FILTERED)
        self.edge = EdgeNode(edge_profile_name, ml_filtering_enabled, self._rng)
        self.cloud = CloudNode(self._rng)

        self.time = 0.0
        self.distance = 0.0

        self.total_cloud_bits_sent = 0.0
        self.total_cloud_bits_attempted = 0.0
        self.total_cloud_bits_raw_equivalent = 0.0

        self.metrics: list[SimMetrics] = []

    def _get_segment(self):
        d = 0.0
        for seg in self.route:
            d += seg.length_m
            if self.distance <= d:
                return seg
        return self.route[-1]

    @staticmethod
    def _classify_state(edge_lat, edge_plr, cloud_lat, cloud_plr) -> str:
        edge_ok = (edge_lat < config.LAT_SAFETY_THRESHOLD_S) and (edge_plr < config.PLR_SAFETY_THRESHOLD)
        cloud_ok = (cloud_lat < config.LAT_SAFETY_THRESHOLD_S) and (cloud_plr < config.PLR_SAFETY_THRESHOLD)

        if edge_ok and cloud_ok:
            return 'Both OK'
        if edge_ok and not cloud_ok:
            return 'Cloud Down'
        if (not edge_ok) and cloud_ok:
            return 'Edge Down'
        return 'Both Down'

    def _split_cloud_input(self, safety_bits: float, monitoring_bits: float, anomaly_active: bool):
        """
        Determines the data volume actually forwarded to the Cloud, depending
        on the architectural mode (see models.ArchitectureMode).
        """
        if self.architecture_mode == ArchitectureMode.PARALLEL_RAW:
            return safety_bits, monitoring_bits

        # HYBRID_FILTERED: safety always fully; monitoring — compressed,
        # except for active anomaly periods (full raw stream for
        # dispatcher verification).
        cloud_safety = safety_bits
        if anomaly_active:
            cloud_monitoring = monitoring_bits
        else:
            cloud_monitoring = monitoring_bits * config.EDGE_TO_CLOUD_COMPRESSION_RATIO
        return cloud_safety, cloud_monitoring

    def step(self, dt: float) -> None:
        seg = self._get_segment()

        self.distance += (seg.speed_kmh / 3.6) * dt
        self.time += dt

        anomaly_active = self.anomaly.step(dt)
        sample = self.traffic_gen.generate(dt, anomaly_active)

        # --- Edge: always analyzes the full stream ---
        edge_result = self.edge.step(self.time, dt, sample.safety_bits, sample.monitoring_bits, anomaly_active)

        # --- determining the volume going to the Cloud based on the architectural mode ---
        cloud_safety_bits, cloud_monitoring_bits = self._split_cloud_input(
            sample.safety_bits, sample.monitoring_bits, anomaly_active
        )
        self.total_cloud_bits_raw_equivalent += sample.safety_bits + sample.monitoring_bits

        # --- radio channel ---
        channel_sample = self.channel.step(seg, self.distance, dt)

        # --- Cloud: network + server segment ---
        cloud_result = self.cloud.step(
            self.time, dt, cloud_safety_bits, cloud_monitoring_bits,
            channel_sample.capacity_bps, channel_sample.harq_extra_latency_s,
        )
        self.total_cloud_bits_sent += cloud_result.bits_transmitted
        self.total_cloud_bits_attempted += cloud_result.bits_attempted

        # --- system state classification (13) ---
        state = self._classify_state(
            edge_result.latency, edge_result.plr,
            cloud_result.total_latency, cloud_result.plr,
        )

        # --- economics ---
        cumulative_cost_usd = cost_model.total_cost_usd(
            edge_result.energy_cumulative_j,
            cloud_result.tx_energy_cumulative_j,
            self.total_cloud_bits_sent,
        )

        self.metrics.append(SimMetrics(
            run_id=self.run_id,
            scenario=self.scenario_name,
            edge_profile=self.edge_profile_name,
            architecture_mode=self.architecture_mode.value,
            anomaly_on=self.anomaly.enabled,
            timestamp=self.time,
            distance=self.distance,
            speed=seg.speed_kmh,
            env_type=seg.env_type.name,

            edge_queue_bits=edge_result.queue_bits,
            edge_temp=edge_result.temperature,
            edge_latency=edge_result.latency,
            edge_plr=edge_result.plr,
            edge_aoi=edge_result.aoi,
            edge_safety_aoi=edge_result.safety_aoi,
            edge_monitoring_aoi=edge_result.monitoring_aoi,
            edge_safety_plr=edge_result.safety_plr,
            edge_monitoring_plr=edge_result.monitoring_plr,
            edge_energy_j=edge_result.energy_j,

            cloud_queue_bits=cloud_result.queue_bits,
            cloud_capacity=channel_sample.capacity_bps,
            cloud_harq_bler=channel_sample.bler,
            cloud_latency=cloud_result.total_latency,
            cloud_plr=cloud_result.plr,
            cloud_aoi=cloud_result.aoi,
            cloud_safety_aoi=cloud_result.safety_aoi,
            cloud_monitoring_aoi=cloud_result.monitoring_aoi,
            cloud_safety_plr=cloud_result.safety_plr,
            cloud_monitoring_plr=cloud_result.monitoring_plr,

            cloud_server_latency=cloud_result.server_latency,

            cloud_bits_sent=self.total_cloud_bits_sent,
            cloud_bits_attempted=self.total_cloud_bits_attempted,
            cloud_bits_raw_equivalent=self.total_cloud_bits_raw_equivalent,
            edge_energy_cumulative_j=edge_result.energy_cumulative_j,
            cloud_tx_energy_cumulative_j=cloud_result.tx_energy_cumulative_j,
            cumulative_cost_usd=cumulative_cost_usd,

            anomaly_active=anomaly_active,
            system_state=state,
        ))

    def run(self) -> None:
        """Executes the run to the end of the route (convenience wrapper method)."""
        total_len = sum(s.length_m for s in self.route)
        while self.distance < total_len:
            self.step(config.SIM_TICK_S)