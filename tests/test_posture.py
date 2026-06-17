"""Testes do sinal de POSTURA de uso de celular (independe da caixa COCO).

A postura é estimada a partir dos 17 keypoints COCO. Montamos esqueletos
sintéticos (calibrados pelo vídeo real de sala de aula) representando:

  - **uso**: as duas mãos juntas e BAIXAS à frente do corpo (segurando o
    aparelho no colo/à frente), cotovelos flexionados e cabeça inclinada para
    baixo (olhando a tela) — pega inclusive o celular escondido;
  - **repouso**: braço esticado para baixo ao lado do corpo, cabeça ereta;
  - **notebook**: mãos afastadas na altura da mesa (teclado). A postura é
    parecida com a do celular — a separação é por CONTEXTO (supressão por
    notebook em ``associate``), validada aqui.

Os testes não usam YOLO real — só a geometria pura ``phone_use_posture`` e a
regra ``Detector.associate`` (com o sinal autônomo de postura).
"""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.detector import (
    KP_L_ELBOW,
    KP_L_EYE,
    KP_L_HIP,
    KP_L_SHOULDER,
    KP_L_WRIST,
    KP_NOSE,
    KP_R_EYE,
    KP_R_HIP,
    KP_R_SHOULDER,
    KP_R_ELBOW,
    KP_R_WRIST,
    Detection,
    Detector,
    PersonDetection,
    phone_use_posture,
)

CFG = Config()


def _base_skeleton() -> np.ndarray:
    """Esqueleto base com ombros, quadris, olhos e nariz erető (confiança alta)."""
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
    """Duas mãos juntas, baixas à frente do corpo + cotovelos dobrados + cabeça baixa."""
    kp = _base_skeleton()
    # Cabeça baixa: nariz cai bem abaixo da linha dos olhos.
    kp[KP_NOSE] = (150.0, 100.0, 0.9)
    # Pulsos juntos e BAIXOS (~0.8 da largura de ombros abaixo), centralizados.
    kp[KP_L_WRIST] = (140.0, 185.0, 0.9)
    kp[KP_R_WRIST] = (160.0, 185.0, 0.9)
    # Cotovelos abertos e um pouco acima dos pulsos (antebraços dobrados).
    kp[KP_L_ELBOW] = (120.0, 165.0, 0.9)
    kp[KP_R_ELBOW] = (200.0, 165.0, 0.9)
    return kp


def _resting_skeleton() -> np.ndarray:
    """Braço estendido para baixo, ao lado do corpo; cabeça ereta."""
    kp = _base_skeleton()
    kp[KP_R_ELBOW] = (195.0, 200.0, 0.9)
    kp[KP_R_WRIST] = (200.0, 275.0, 0.9)  # quase na linha do quadril, lateral
    return kp


def _laptop_typing_skeleton() -> np.ndarray:
    """Mãos AFASTADAS na altura da mesa (teclado) + cabeça baixa — usuário de notebook."""
    kp = _base_skeleton()
    kp[KP_NOSE] = (150.0, 100.0, 0.9)  # também olha para baixo (para a tela)
    # Pulsos pouco abaixo dos ombros (altura da mesa) e bem afastados.
    kp[KP_L_WRIST] = (95.0, 150.0, 0.9)
    kp[KP_R_WRIST] = (205.0, 150.0, 0.9)
    kp[KP_L_ELBOW] = (100.0, 165.0, 0.9)
    kp[KP_R_ELBOW] = (200.0, 165.0, 0.9)
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


def test_laptop_suppresses_standalone_posture():
    """Mão sobre um notebook detectado NÃO marca uso de celular (está digitando)."""
    person = PersonDetection(
        class_id=0, confidence=0.9, box=(90, 50, 210, 300),
        keypoints=_using_skeleton(),
    )
    # Notebook posicionado sob os pulsos (~y 185); o pulso cai sobre a caixa.
    laptop = Detection(class_id=CFG.laptop_class_id, confidence=0.85,
                       box=(120.0, 175.0, 190.0, 210.0))
    out = Detector(config=CFG).associate([person], phones=[], laptops=[laptop])
    assert out[0].on_laptop is True
    assert out[0].using_phone is False
    assert out[0].by_posture is False


def test_no_laptop_keeps_standalone_posture():
    """Sem notebook por perto, a mesma postura volta a marcar uso (controle)."""
    person = PersonDetection(
        class_id=0, confidence=0.9, box=(90, 50, 210, 300),
        keypoints=_using_skeleton(),
    )
    far_laptop = Detection(class_id=CFG.laptop_class_id, confidence=0.85,
                          box=(600.0, 175.0, 680.0, 210.0))  # longe dos pulsos
    out = Detector(config=CFG).associate([person], phones=[], laptops=[far_laptop])
    assert out[0].on_laptop is False
    assert out[0].using_phone is True
    assert out[0].by_posture is True


def test_laptop_typing_posture_suppressed_by_context():
    """Usuário de notebook: postura parecida, mas a mão no teclado suprime o uso."""
    person = PersonDetection(
        class_id=0, confidence=0.9, box=(80, 50, 220, 300),
        keypoints=_laptop_typing_skeleton(),
    )
    laptop = Detection(class_id=CFG.laptop_class_id, confidence=0.85,
                       box=(90.0, 140.0, 210.0, 175.0))  # sob as duas mãos
    out = Detector(config=CFG).associate([person], phones=[], laptops=[laptop])
    assert out[0].on_laptop is True
    assert out[0].using_phone is False
