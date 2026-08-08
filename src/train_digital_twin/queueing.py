"""
queueing.py
===========
Reusable packet queue primitives with generation time tracking
(for correct Age of Information calculation) and support for two QoS
priority classes (SAFETY / MONITORING).

Design Decisions
----------------
- Each "data chunk" (packet) is a record [generation_time, remaining_bits]
  appended to the tail of the queue at each simulation tick. This is a
  generalization of the mechanisms previously implemented separately in Cloud
  (TTL reneging) and, after fixing, in Edge (processing time tracking) into
  one common class.
- PacketQueue implements a pure FIFO queue of a single priority class.
- PriorityQueueSystem combines two PacketQueues (SAFETY, MONITORING) with
  strict service priority (SAFETY is always served first) and priority dropping
  upon shared buffer overflow (MONITORING is dropped first).
"""

from collections import deque


class PacketQueue:
    """FIFO packet queue of a single class with generation time tracking."""

    def __init__(self):
        self._packets = deque()   # elements: [gen_time, remaining_bits]
        self.total_bits = 0.0

    def enqueue(self, gen_time: float, bits: float) -> None:
        """Appends a new data chunk to the tail of the queue."""
        if bits <= 0:
            return
        self._packets.append([gen_time, bits])
        self.total_bits += bits

    def serve(self, capacity_bits: float):
        """
        Serves the queue from the head for a volume of `capacity_bits`.

        Returns (fully_delivered_gen_times, bits_served) — a list of generation
        times of packets fully delivered by this call (for AoI), and the
        actually served volume of bits (can be less than capacity_bits if the
        queue emptied earlier).
        """
        remaining_capacity = capacity_bits
        delivered_gen_times = []
        bits_served = 0.0

        while remaining_capacity > 0 and self._packets:
            head = self._packets[0]
            if head[1] <= remaining_capacity:
                remaining_capacity -= head[1]
                bits_served += head[1]
                delivered_gen_times.append(head[0])
                self._packets.popleft()
            else:
                head[1] -= remaining_capacity
                bits_served += remaining_capacity
                remaining_capacity = 0.0

        self.total_bits -= bits_served
        return delivered_gen_times, bits_served

    def drop_expired(self, current_time: float, ttl: float) -> float:
        """
        Removes from the head of the queue packets whose waiting time exceeded ttl.

        Due to the monotonicity of generation time, it is sufficient to check only
        the head: if it is not yet expired, no subsequent (younger) packet is
        expired either.

        Returns the total volume of dropped bits.
        """
        dropped_bits = 0.0
        while self._packets and (current_time - self._packets[0][0] > ttl):
            expired = self._packets.popleft()
            dropped_bits += expired[1]
        self.total_bits -= dropped_bits
        return dropped_bits

    def tail_drop(self, max_total_bits: float) -> float:
        """
        If `total_bits` exceeds `max_total_bits`, drops the excess from the
        TAIL of the queue (newest data) — physically corresponds to dropping
        packets that do not fit in the input buffer.

        Returns the volume of dropped bits (0 if there was no overflow).
        """
        overflow = self.total_bits - max_total_bits
        if overflow <= 0:
            return 0.0

        remaining_to_drop = overflow
        while remaining_to_drop > 0 and self._packets:
            tail = self._packets[-1]
            if tail[1] <= remaining_to_drop:
                remaining_to_drop -= tail[1]
                self._packets.pop()
            else:
                tail[1] -= remaining_to_drop
                remaining_to_drop = 0.0

        self.total_bits = max_total_bits
        return overflow

    def __len__(self):
        return len(self._packets)


class PriorityQueueSystem:
    """
    A system of two priority classes (SAFETY, MONITORING) sharing a single
    buffer of limited capacity.

    Service discipline — strict priority: at each step, the SAFETY queue is
    served first with the full available capacity, and only the remaining
    volume is directed to the MONITORING queue.

    Overflow drop discipline — priority protection of SAFETY: the excess over
    the buffer is first removed from the MONITORING queue (tail-drop), and
    only if MONITORING is already empty but the buffer is still overflowing,
    the excess is removed from SAFETY (this is an extreme, highly unlikely case).
    """

    def __init__(self, buffer_max_bits: float):
        self.buffer_max_bits = buffer_max_bits
        self.safety = PacketQueue()
        self.monitoring = PacketQueue()

    @property
    def total_bits(self) -> float:
        return self.safety.total_bits + self.monitoring.total_bits

    def enqueue(self, gen_time: float, safety_bits: float, monitoring_bits: float) -> None:
        self.safety.enqueue(gen_time, safety_bits)
        self.monitoring.enqueue(gen_time, monitoring_bits)

    def serve(self, capacity_bits: float):
        """
        Serves both queues with strict priority.

        Returns a dictionary detailing by classes:
            {
                'safety': (delivered_gen_times, bits_served),
                'monitoring': (delivered_gen_times, bits_served),
                'total_bits_served': float,
            }
        """
        safety_delivered, safety_served = self.safety.serve(capacity_bits)
        remaining = capacity_bits - safety_served
        monitoring_delivered, monitoring_served = self.monitoring.serve(max(remaining, 0.0))

        return {
            'safety': (safety_delivered, safety_served),
            'monitoring': (monitoring_delivered, monitoring_served),
            'total_bits_served': safety_served + monitoring_served,
        }

    def drop_expired(self, current_time: float, ttl: float):
        """Applies TTL reneging to both classes. Returns (safety_dropped, monitoring_dropped)."""
        return (
            self.safety.drop_expired(current_time, ttl),
            self.monitoring.drop_expired(current_time, ttl),
        )

    def enforce_buffer(self):
        """
        Ensures total_bits <= buffer_max_bits, dropping first from MONITORING,
        and if necessary, from SAFETY. Returns (safety_dropped, monitoring_dropped).
        """
        overflow = self.total_bits - self.buffer_max_bits
        if overflow <= 0:
            return 0.0, 0.0

        monitoring_dropped = self.monitoring.tail_drop(
            max(self.monitoring.total_bits - overflow, 0.0)
        )
        remaining_overflow = overflow - monitoring_dropped
        safety_dropped = 0.0
        if remaining_overflow > 0:
            safety_dropped = self.safety.tail_drop(
                max(self.safety.total_bits - remaining_overflow, 0.0)
            )
        return safety_dropped, monitoring_dropped