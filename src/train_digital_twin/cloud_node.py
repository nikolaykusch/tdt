"""
cloud_node.py
=============
Cloud subsystem model consisting of TWO sequential segments:

1. Network stage — transmission over a radio channel with
   a TTL deadline queue (reneging), just like in the previous model version,
   but now with support for two QoS priority classes.

2. Server stage — NEW realism element:
   models finite backend throughput (multi-tenant load from parallel trains)
   and fixed processing delay (message queue/DB write). Previously, this
   segment was absent: all Cloud latency was reduced solely to the channel,
   neglecting the server queue and processing.

Radio transmitter energy (RADIO_TX_POWER_W) is accumulated right here, since
it is the network segment that activates the radio modem.
"""

import random
from dataclasses import dataclass

from . import config
from .queueing import PriorityQueueSystem


@dataclass
class CloudStepResult:
    network_latency: float
    server_latency: float
    total_latency: float
    plr: float
    safety_plr: float
    monitoring_plr: float
    aoi: float
    safety_aoi: float
    monitoring_aoi: float
    queue_bits: float
    bits_attempted: float            # volume Edge decided to send BEFORE the channel (for traffic savings metric)
    bits_transmitted: float          # volume actually broadcasted after the channel (for traffic billing)
    tx_energy_j: float
    tx_energy_cumulative_j: float


class CloudNode:
    """Cloud subsystem state (network + server segment) for a single run."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()

        self.network_queues = PriorityQueueSystem(buffer_max_bits=float('inf'))
        self.server_queues = PriorityQueueSystem(buffer_max_bits=float('inf'))

        self.total_generated = 0.0
        self.total_safety_generated = 0.0
        self.total_monitoring_generated = 0.0
        self.total_dropped = 0.0
        self.total_safety_dropped = 0.0
        self.total_monitoring_dropped = 0.0

        self.last_delivered_gen_time_safety = 0.0
        self.last_delivered_gen_time_monitoring = 0.0
        self.tx_energy_cumulative_j = 0.0

    def step(self, t: float, dt: float, safety_bits: float, monitoring_bits: float,
             capacity_bps: float, harq_extra_latency_s: float) -> CloudStepResult:
        # ------------------------------------------------------------------
        # 1. Network segment: channel transmission with TTL deadline
        # ------------------------------------------------------------------
        self.network_queues.enqueue(t, safety_bits, monitoring_bits)
        self.total_generated += safety_bits + monitoring_bits
        self.total_safety_generated += safety_bits
        self.total_monitoring_generated += monitoring_bits
        bits_attempted = safety_bits + monitoring_bits

        transmitted_capacity = capacity_bps * dt
        net_serve = self.network_queues.serve(transmitted_capacity)
        bits_transmitted = net_serve['total_bits_served']

        net_safety_dropped, net_monitoring_dropped = self.network_queues.drop_expired(t, config.TTL_SECONDS)
        self.total_dropped += net_safety_dropped + net_monitoring_dropped
        self.total_safety_dropped += net_safety_dropped
        self.total_monitoring_dropped += net_monitoring_dropped

        network_latency = (self.network_queues.total_bits / capacity_bps
                            if capacity_bps > 0 else config.TTL_SECONDS)
        network_latency += harq_extra_latency_s

        # ------------------------------------------------------------------
        # 2. Server segment: finite backend throughput +
        #    fixed processing delay (NEW model element)
        # ------------------------------------------------------------------
        safety_delivered_net, safety_bits_net = net_serve['safety']
        monitoring_delivered_net, monitoring_bits_net = net_serve['monitoring']

        self.server_queues.enqueue(t, safety_bits_net, monitoring_bits_net)
        server_capacity = config.CLOUD_SERVER_CAPACITY_BPS * dt
        server_serve = self.server_queues.serve(server_capacity)

        server_latency = (self.server_queues.total_bits / config.CLOUD_SERVER_CAPACITY_BPS
                          if config.CLOUD_SERVER_CAPACITY_BPS > 0 else 0.0)
        server_latency += config.CLOUD_SERVER_FIXED_DELAY_S

        total_latency = network_latency + server_latency

        # --- AoI: separately per class, based on packets that passed BOTH segments ---
        safety_delivered_srv, monitoring_delivered_srv = server_serve['safety'][0], server_serve['monitoring'][0]
        if safety_delivered_srv:
            self.last_delivered_gen_time_safety = max(safety_delivered_srv)
        if monitoring_delivered_srv:
            self.last_delivered_gen_time_monitoring = max(monitoring_delivered_srv)

        safety_aoi = t - self.last_delivered_gen_time_safety
        monitoring_aoi = t - self.last_delivered_gen_time_monitoring
        aoi = max(safety_aoi, monitoring_aoi)

        # --- final quality metrics (determined by the network segment,
        #     since real data loss occurs there; the server segment is
        #     considered reliable (lossless) and only adds delay) ---
        plr = self.total_dropped / self.total_generated if self.total_generated > 0 else 0.0
        safety_plr = (self.total_safety_dropped / self.total_safety_generated
                      if self.total_safety_generated > 0 else 0.0)
        monitoring_plr = (self.total_monitoring_dropped / self.total_monitoring_generated
                           if self.total_monitoring_generated > 0 else 0.0)

        # --- radio transmitter energy: consumed as long as the channel is active
        #     (even if there is nothing to transmit, the transceiver remains
        #     in an active idle state) ---
        tx_energy_j = config.RADIO_TX_POWER_W * dt if capacity_bps > 0 else 0.0
        self.tx_energy_cumulative_j += tx_energy_j

        return CloudStepResult(
            network_latency=network_latency,
            server_latency=server_latency,
            total_latency=total_latency,
            plr=plr,
            safety_plr=safety_plr,
            monitoring_plr=monitoring_plr,
            aoi=aoi,
            safety_aoi=safety_aoi,
            monitoring_aoi=monitoring_aoi,
            queue_bits=self.network_queues.total_bits + self.server_queues.total_bits,
            bits_attempted=bits_attempted,
            bits_transmitted=bits_transmitted,
            tx_energy_j=tx_energy_j,
            tx_energy_cumulative_j=self.tx_energy_cumulative_j,
        )