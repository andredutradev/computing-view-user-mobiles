"""Testes do sinal de POSTURA de uso de celular (independe da caixa COCO).

A postura é estimada a partir dos 17 keypoints COCO. Montamos esqueletos
sintéticos representando duas situações canônicas:

  - **uso**: mão erguida à frente do tronco + cotovelo flexionado + cabeça
    inclinada para baixo (nariz bem abaixo da linha dos olhos);
  - **repouso**: braço esticado para baixo ao lado do corpo, cabeça ereta.

Os testes não usam YOLO real — só a geometria pura ``phone_use_posture`` e a
regra ``Detector.associate`` (com o sinal autônomo de postura).
"""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.detector import (
    KP_L_EYE,
    KP_L_HIP,
    KP_L_SHOULDER,
    KP_NOSE,
    KP_R_EYE,
    KP_R_HIP,
    KP_R_SHOULDER,
    KP_R_ELBOW,
    KP_R_WRIST,
    Detector,
    PersonDetection,
    phone_use_posture,
)

CFG = Config()


def _base_skeleton() -> np.ndarray:
    """Esqueleto base com ombros, quadris, olhos e nariz erетo (confiança alta)."""
    kp = np.zeros((17, 3), dtype=np.float32)
    # Ombros (largura 80px) e quadris ~160px abaixo.
    kp[KP_L_SHOULDER] = (110.0, 120.0, 0.95)
    kp[KP_R_SHOULDER] = (190.0, 120.0, 0.95)
    kp[KP_L_HIP] = (120.0, 280.0, 0.9)
    kp[KP_R_HIP] = (180.0, 280.0, 0.9)
    # Cabeça ereta: olhos acima do nariz, nariz pouco abaixo dos olhos.
    kp[KP_L_EYE] = (140.0, 70.0, 0.9)
    kp[KP_R_EYE] = (160.0, 70.0, 0.9)
    kp[KP_NOSE] = (150.0, 78.0, 0.9)  # ~0.1 da largura de ombros abaixo
    return kp


def _using_skeleton() -> np.ndarray:
    """Mão direita erguida ao peito + cotovelo dobrado + cabeça baixa."""
    kp = _base_skeleton()
    # Cabeça baixa: nariz cai bem abaixo da linha dos olhos.
    kp[KP_NOSE] = (150.0, 100.0, 0.9)
    # Antebraço subindo: cotovelo abaixo, pulso trazido ao centro/peito.
    kp[KP_R_ELBOW] = (185.0, 200.0, 0.9)
    kp[KP_R_WRIST] = (150.0, 150.0, 0.9)  # centralizado, acima do quadril
    return kp


def _resting_skeleton() -> np.ndarray:
    """Braço estendido para baixo, ao lado do corpo; cabeça ereta."""
    kp = _base_skeleton()
    kp[KP_R_ELBOW] = (195.0, 200.0, 0.9)
    kp[KP_R_WRIST] = (200.0, 275.0, 0.9)  # quase na linha do quadril, lateral
    return kp


def test_posture_high_when_using():
    score = phone_use_posture(_using_skeleton(), (90, 50, 210, 300), CFG)
    assert score >= 0.6, f"esperava postura alta, veio {score:.2f}"


def test_posture_low_when_resting():
    score = phone_use_posture(_resting_skeleton(), (90, 50, 210, 300), CFG)
    assert score <= 0.3, f"esperava postura baixa, veio {score:.2f}"


def test_posture_zero_without_shoulders():
    kp = np.zeros((17, 3), dtype=np.float32)  # nada confiável
    assert phone_use_posture(kp, (0, 0, 100, 200), CFG) == 0.0


def test_posture_zero_without_keypoints():
    assert phone_use_posture(None, (0, 0, 100, 200), CFG) == 0.0


def test_standalone_posture_flags_using_without_phone():
    """Sem nenhuma caixa de celular, a postura forte sozinha marca uso."""
    person = PersonDetection(
        class_id=0, confidence=0.9, box=(90, 50, 210, 300),
        keypoints=_using_skeleton(),
    )
    det = Detector(config=CFG)
    out = det.associate([person], phones=[])
    assert out[0].using_phone is True
    assert out[0].by_posture is True
    assert out[0].matched_phone is None


def test_resting_posture_not_flagged_without_phone():
    person = PersonDetection(
        class_id=0, confidence=0.9, box=(90, 50, 210, 300),
        keypoints=_resting_skeleton(),
    )
    det = Detector(config=CFG)
    out = det.associate([person], phones=[])
    assert out[0].using_phone is False
    assert out[0].by_posture is False


def test_posture_disabled_score_zero_and_no_flag():
    cfg = Config(posture_enabled=False)
    person = PersonDetection(
        class_id=0, confidence=0.9, box=(90, 50, 210, 300),
        keypoints=_using_skeleton(),
    )
    out = Detector(config=cfg).associate([person], phones=[])
    assert out[0].posture_score == 0.0
    assert out[0].using_phone is False
