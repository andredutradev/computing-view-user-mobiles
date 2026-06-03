"""Testes do modo demo (cena sintética + detector roteirizado).

Validam, sem GUI nem YOLO, que: a fonte sintética produz frames; o detector
roteirizado alterna o estado "usando celular" conforme o celular sobrepõe a
pessoa; e o pipeline completo roda headless.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import settings
from src.demo import (
    CYCLE,
    ScriptedDetector,
    SyntheticVideoSource,
    demo_scene,
)
from src.detector import KP_R_WRIST
from src.visualizer import Visualizer


def test_synthetic_source_produces_frames():
    src = SyntheticVideoSource(max_frames=3)
    frames = list(src.frames())
    assert len(frames) == 3
    assert all(isinstance(f, np.ndarray) for f in frames)
    assert frames[0].shape == (480, 640, 3)


def test_demo_scene_has_phase_with_and_without_phone():
    # Em algum frame do ciclo o celular existe; em outro, não.
    phones = [demo_scene(i)[2] for i in range(CYCLE)]
    assert any(p is not None for p in phones)
    assert any(p is None for p in phones)


def test_demo_scene_provides_keypoints():
    # Toda cena traz keypoints (17, 3); o pulso direito é confiável.
    _, keypoints, _ = demo_scene(0)
    assert keypoints.shape == (17, 3)
    assert keypoints[KP_R_WRIST][2] > 0.3


def test_demo_wrist_follows_phone_when_present():
    # No frame com celular, o pulso direito pousa sobre o aparelho.
    for i in range(CYCLE):
        _, keypoints, phone = demo_scene(i)
        if phone is not None:
            px = (phone[0] + phone[2]) / 2.0
            py = (phone[1] + phone[3]) / 2.0
            assert keypoints[KP_R_WRIST][0] == pytest.approx(px)
            assert keypoints[KP_R_WRIST][1] == pytest.approx(py)
            break


def test_scripted_detector_toggles_using_phone():
    detector = ScriptedDetector(config=settings)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    using_states = []
    for _ in range(CYCLE):
        people = detector.process_frame(frame)
        assert len(people) == 1  # sempre há exatamente uma pessoa
        using_states.append(people[0].using_phone)

    # Durante o ciclo, a pessoa deve passar por AMBOS os estados.
    assert any(using_states), "esperava ao menos um frame 'usando celular'"
    assert any(not s for s in using_states), "esperava ao menos um 'sem celular'"


def test_demo_pipeline_headless_runs():
    src = SyntheticVideoSource(max_frames=CYCLE)
    detector = ScriptedDetector(config=settings)
    vis = Visualizer(config=settings)

    saw_green_state = False
    for frame in src.frames():
        people = detector.process_frame(frame)
        phones = [p.matched_phone for p in people if p.matched_phone]
        annotated = vis.draw(frame, people, phones)
        assert annotated.shape == frame.shape
        if any(p.using_phone for p in people):
            saw_green_state = True
    assert saw_green_state
