"""Tests for queueing.PacketQueue and queueing.PriorityQueueSystem."""

import pytest

from train_digital_twin.queueing import PacketQueue, PriorityQueueSystem


def test_packet_queue_basic_fifo_serve():
    q = PacketQueue()
    q.enqueue(0.0, 100.0)
    q.enqueue(0.01, 50.0)

    delivered, served = q.serve(120.0)
    assert served == 120.0
    # the first packet (100 bits) is fully delivered; the second is partially consumed
    assert delivered == [0.0]
    assert q.total_bits == pytest.approx(30.0)


def test_packet_queue_full_drain_marks_all_delivered():
    q = PacketQueue()
    q.enqueue(0.0, 10.0)
    q.enqueue(0.01, 10.0)
    delivered, served = q.serve(1000.0)
    assert served == 20.0
    assert delivered == [0.0, 0.01]
    assert q.total_bits == 0.0
    assert len(q) == 0


def test_packet_queue_ttl_reneging_only_checks_head():
    q = PacketQueue()
    q.enqueue(0.0, 10.0)
    q.enqueue(1.0, 10.0)
    # At t=2.0 with TTL=1.5, the first packet (age 2.0) is expired,
    # the second (age 1.0) — not yet.
    dropped = q.drop_expired(current_time=2.0, ttl=1.5)
    assert dropped == 10.0
    assert q.total_bits == 10.0
    assert len(q) == 1


def test_packet_queue_tail_drop_removes_newest_first():
    q = PacketQueue()
    q.enqueue(0.0, 100.0)   # old packet
    q.enqueue(0.01, 50.0)   # new packet
    dropped = q.tail_drop(max_total_bits=120.0)
    assert dropped == pytest.approx(30.0)
    # the old packet should remain untouched, the new one — truncated
    delivered, served = q.serve(1000.0)
    assert served == pytest.approx(120.0)
    assert delivered == [0.0, 0.01]


def test_priority_queue_strict_priority_serves_safety_first():
    system = PriorityQueueSystem(buffer_max_bits=float('inf'))
    system.enqueue(0.0, safety_bits=50.0, monitoring_bits=100.0)

    result = system.serve(capacity_bits=70.0)
    # SAFETY (50) is fully served, the remainder (20) goes to MONITORING
    assert result['safety'][1] == 50.0
    assert result['monitoring'][1] == 20.0
    assert system.monitoring.total_bits == pytest.approx(80.0)
    assert system.safety.total_bits == 0.0


def test_priority_queue_buffer_enforcement_drops_monitoring_first():
    system = PriorityQueueSystem(buffer_max_bits=100.0)
    system.enqueue(0.0, safety_bits=60.0, monitoring_bits=80.0)  # total 140 > 100

    safety_dropped, monitoring_dropped = system.enforce_buffer()
    assert safety_dropped == 0.0
    assert monitoring_dropped == pytest.approx(40.0)
    assert system.total_bits == pytest.approx(100.0)
    assert system.safety.total_bits == pytest.approx(60.0)


def test_priority_queue_buffer_enforcement_touches_safety_only_as_last_resort():
    system = PriorityQueueSystem(buffer_max_bits=30.0)
    system.enqueue(0.0, safety_bits=60.0, monitoring_bits=20.0)  # total 80 > 30

    safety_dropped, monitoring_dropped = system.enforce_buffer()
    # MONITORING (20) is dropped completely, but we still need to free up
    # another 30 -> these 30 are removed from SAFETY
    assert monitoring_dropped == pytest.approx(20.0)
    assert safety_dropped == pytest.approx(30.0)
    assert system.total_bits == pytest.approx(30.0)