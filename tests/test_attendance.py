"""Testes do AttendanceTracker (presença/identidade/ocupação) com mocks."""

from __future__ import annotations

import numpy as np

from src.attendance.attendance import AttendanceTracker, MovementState
from src.attendance.enrollment import Gallery, StudentRecord
from src.config import Config
from src.detector import PersonDetection


class _FakeRecognizer:
    """Sempre 'reconhece' o mesmo aluno (sem InsightFace)."""

    def __init__(self, student_id="ana", score=0.9):
        self.student_id = student_id
        self.score = score

    def embed_face(self, crop):
        return np.ones(4, dtype=np.float32)

    def match(self, emb, gallery, threshold):
        return (self.student_id, self.score)


def _gallery():
    return Gallery(
        {"ana": StudentRecord("ana", "Ana Silva", np.ones((1, 4), dtype=np.float32), (), "")}
    )


def _person(track_id, box=(100.0, 50.0, 200.0, 400.0), using=False):
    return PersonDetection(
        class_id=0,
        confidence=0.9,
        box=box,
        using_phone=using,
        keypoints=None,
        track_id=track_id,
    )


def _cfg():
    return Config(
        face_recog_every_n_frames=1,
        face_confirm_hits=2,
        face_confirm_score=0.99,
        face_match_threshold=0.35,
        presence_grace_seconds=1.0,
        attendance_min_seconds=0.0,
    )


def test_pre_registers_enrolled_students():
    tracker = AttendanceTracker(config=_cfg(), gallery=_gallery(), recognizer=_FakeRecognizer(), fps=10.0)
    assert "ana" in tracker.session.students  # aparece mesmo antes de ser vista


def test_identifies_and_credits_presence():
    tracker = AttendanceTracker(
        config=_cfg(), gallery=_gallery(), recognizer=_FakeRecognizer(), fps=10.0
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):
        tracker.update(frame, [_person(1)], i)

    assert tracker.identity.identity_of(1) == "ana"
    assert tracker.identity_labels().get(1) == "Ana Silva"
    session = tracker.finalize()
    # Presente do frame 0 (t=0) ao 4 (t=0.4) -> ~0.4s acumulados.
    assert session.students["ana"].present_seconds > 0.0
    assert len(session.occupancy) == 5


def test_credits_phone_time_to_identified_student():
    tracker = AttendanceTracker(
        config=_cfg(), gallery=_gallery(), recognizer=_FakeRecognizer(), fps=10.0
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):  # 5 frames usando celular -> ~0.4s
        tracker.update(frame, [_person(1, using=True)], i)
    session = tracker.finalize()
    assert session.students["ana"].phone_seconds > 0.0


def test_credits_phone_time_to_anonymous_track():
    # Sem galeria/reconhecedor: o track tem id mas nunca é identificado.
    tracker = AttendanceTracker(config=_cfg(), gallery=None, recognizer=None, fps=10.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(4):
        tracker.update(frame, [_person(7, using=True)], i)
    session = tracker.finalize()
    assert "anon_7" in session.students
    assert session.students["anon_7"].display_name == "Pessoa #7"
    assert session.students["anon_7"].phone_seconds > 0.0


def test_occupancy_counts_untracked_people():
    tracker = AttendanceTracker(config=_cfg(), gallery=_gallery(), recognizer=_FakeRecognizer(), fps=10.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Uma pessoa sem track_id (cold-start) + uma usando celular.
    tracker.update(frame, [_person(None), _person(2, using=True)], 0)
    sample = tracker.session.occupancy[0]
    assert sample.people_count == 2
    assert sample.using_phone_count == 1


def test_movement_seated_when_still():
    cfg = Config(seated_min_frames=3, seated_speed_threshold=0.02, move_speed_threshold=0.06)
    tracker = AttendanceTracker(config=cfg, gallery=None, recognizer=None, fps=10.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):
        tracker.update(frame, [_person(1)], i)  # caixa imóvel
    assert tracker._tracks[1].movement == MovementState.SEATED


def test_recognize_small_head_roi_uses_fallback(tmp_path):
    """Regressão: caixa pequena força o fallback do recorte sem quebrar.

    Com a caixa minúscula, o ROI da cabeça fica abaixo de face_min_size_px e
    o código cai no fallback (caixa inteira). Antes da correção, o ``or`` sobre
    um array numpy levantava ValueError aqui.
    """
    cfg = Config(face_min_size_px=40, face_recog_every_n_frames=1)
    tracker = AttendanceTracker(
        config=cfg, gallery=_gallery(), recognizer=_FakeRecognizer(), fps=10.0
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tiny = _person(1, box=(10.0, 10.0, 28.0, 40.0))  # ROI da cabeça < 40px
    tracker.update(frame, [tiny], 0)  # não deve levantar
    assert tracker.identity.identity_of(1) == "ana"


def test_works_without_recognizer_or_gallery():
    """Sem reconhecedor/galeria, ainda conta ocupação e movimentação."""
    tracker = AttendanceTracker(config=_cfg(), gallery=None, recognizer=None, fps=10.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.update(frame, [_person(1)], 0)
    assert len(tracker.session.occupancy) == 1
    assert tracker.identity_labels() == {}
