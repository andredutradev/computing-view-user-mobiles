"""Testes do modelo de sessão (dwell time + ocupação)."""

from __future__ import annotations

import pytest

from src.attendance.session import OccupancySample, Session, StudentSession


def test_dwell_bridges_short_gap():
    st = StudentSession("a", "A")
    st.mark_present(0.0, grace_seconds=3.0)
    st.mark_present(2.0, grace_seconds=3.0)  # gap 2 <= 3 -> ponte
    assert st.present_seconds == pytest.approx(2.0)
    assert st.num_intervals == 1


def test_dwell_splits_long_gap_and_reentry():
    st = StudentSession("a", "A")
    st.mark_present(0.0, 3.0)
    st.mark_present(1.0, 3.0)
    st.mark_present(10.0, 3.0)  # gap 9 > 3 -> fecha e reabre (re-entrada)
    st.finalize()
    assert st.num_intervals == 2
    # 1o intervalo: 0->1 (1s); 2o: 10->10 (0s).
    assert st.present_seconds == pytest.approx(1.0)
    assert st.first_seen_t == pytest.approx(0.0)
    assert st.last_seen_t == pytest.approx(10.0)


def test_present_seconds_open_interval_counts():
    st = StudentSession("a", "A")
    st.mark_present(5.0, 3.0)
    st.mark_present(8.0, 3.0)  # ainda aberto (gap 3 <= 3)
    assert st.present_seconds == pytest.approx(3.0)


def test_phone_seconds_accumulates_with_bridge_and_reentry():
    st = StudentSession("a", "A")
    st.mark_phone_use(0.0, grace_seconds=3.0)
    st.mark_phone_use(2.0, grace_seconds=3.0)  # gap 2 <= 3 -> ponte (2s)
    st.mark_phone_use(10.0, grace_seconds=3.0)  # gap 9 > 3 -> novo intervalo
    st.finalize()
    # 1o intervalo 0->2 (2s); 2o 10->10 (0s). Independe da presença.
    assert st.phone_seconds == pytest.approx(2.0)
    assert st.present_seconds == 0.0


def test_phone_seconds_open_interval_counts():
    st = StudentSession("a", "A")
    st.mark_phone_use(5.0, 3.0)
    st.mark_phone_use(7.0, 3.0)  # aberto (gap 2 <= 3)
    assert st.phone_seconds == pytest.approx(2.0)


def test_session_mark_phone_use_creates_student():
    s = Session(fps=25.0)
    s.mark_phone_use("anon_3", "Pessoa #3", 0.0, 3.0)
    s.mark_phone_use("anon_3", "Pessoa #3", 1.0, 3.0)
    s.finalize()
    assert s.students["anon_3"].phone_seconds == pytest.approx(1.0)


def test_session_peak_and_average_and_duration():
    s = Session(fps=10.0, source_label="aula")
    s.add_occupancy(OccupancySample(0, 0.0, 2, 1, 0))
    s.add_occupancy(OccupancySample(1, 0.1, 5, 2, 1))
    assert s.peak_occupancy() == (pytest.approx(0.1), 5)
    assert s.average_occupancy() == pytest.approx(3.5)
    assert s.duration_seconds == pytest.approx(0.1)


def test_ensure_student_is_idempotent():
    s = Session(fps=25.0)
    a = s.ensure_student("x", "X")
    b = s.ensure_student("x", "X")
    assert a is b
    assert len(s.students) == 1
