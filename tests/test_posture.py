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
    KP_L_EAR,
    KP_L_ELBOW,
    KP_L_EYE,
    KP_L_HIP,
    KP_L_SHOULDER,
    KP_L_WRIST,
    KP_NOSE,
    KP_R_EAR,
    KP_R_EYE,
    KP_R_HIP,
    KP_R_SHOULDER,
    KP_R_ELBOW,
    KP_R_WRIST,
    Detection,
    Detector,
    PersonDetection,
    face_toward_hands,
    phone_use_posture,
    _is_profile_view,
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


def _hidden_hands_skeleton(nose_drop: float = 40.0) -> np.ndarray:
    """Braços projetados p/ baixo + mãos fundas (escondidas) + rosto baixo.

    ``nose_drop`` = quantos px o nariz cai abaixo das orelhas (quanto maior,
    mais a cabeça está inclinada para baixo). Simula o celular no colo/atrás da
    cadeira, fora de vista.
    """
    kp = _base_skeleton()
    # Orelhas na linha ~70 (como os olhos); nariz cai abaixo delas (rosto baixo).
    kp[KP_L_EAR] = (135.0, 70.0, 0.9)
    kp[KP_R_EAR] = (165.0, 70.0, 0.9)
    kp[KP_NOSE] = (150.0, 70.0 + nose_drop, 0.9)
    # Mãos fundas e centralizadas (na altura do colo, bem abaixo dos ombros).
    kp[KP_L_WRIST] = (140.0, 230.0, 0.9)
    kp[KP_R_WRIST] = (160.0, 230.0, 0.9)
    # Cotovelos abaixo dos ombros e acima dos pulsos (antebraços p/ baixo).
    kp[KP_L_ELBOW] = (125.0, 180.0, 0.9)
    kp[KP_R_ELBOW] = (195.0, 180.0, 0.9)
    return kp


def test_hidden_hands_posture_flags_when_face_down():
    """Mãos escondidas + ROSTO BAIXO -> uso por postura (celular fora de vista)."""
    score = phone_use_posture(_hidden_hands_skeleton(nose_drop=40.0), (90, 50, 210, 320), CFG)
    assert score >= CFG.posture_standalone_threshold, f"esperava alto, veio {score:.2f}"


def _hidden_hands_apart_skeleton(nose_drop: float = 40.0) -> np.ndarray:
    """Cabeça baixa + braços projetados, mas mãos SEPARADAS (à mostra, livres)."""
    kp = _base_skeleton()
    kp[KP_L_EAR] = (135.0, 70.0, 0.9)
    kp[KP_R_EAR] = (165.0, 70.0, 0.9)
    kp[KP_NOSE] = (150.0, 70.0 + nose_drop, 0.9)
    # Pulsos fundos, mas bem afastados um do outro (não estão segurando um objeto).
    kp[KP_L_WRIST] = (95.0, 230.0, 0.9)
    kp[KP_R_WRIST] = (205.0, 230.0, 0.9)
    kp[KP_L_ELBOW] = (100.0, 180.0, 0.9)
    kp[KP_R_ELBOW] = (200.0, 180.0, 0.9)
    return kp


def test_hidden_hands_apart_not_flagged():
    """Mãos à mostra SEPARADAS + cabeça baixa + sem objeto -> NÃO é uso de celular."""
    score = phone_use_posture(_hidden_hands_apart_skeleton(), (80, 50, 220, 320), CFG)
    assert score < CFG.posture_standalone_threshold, f"esperava baixo, veio {score:.2f}"


def test_hidden_hands_require_visible_wrists():
    """Pulso oculto (baixa confiança) -> não valida por postura (cai na via do objeto)."""
    kp = _hidden_hands_skeleton(nose_drop=40.0)
    kp[KP_R_WRIST] = (kp[KP_R_WRIST][0], kp[KP_R_WRIST][1], 0.1)  # mão escondida
    score = phone_use_posture(kp, (90, 50, 210, 320), CFG)
    assert score < CFG.posture_standalone_threshold, f"esperava baixo, veio {score:.2f}"


def test_hidden_hands_posture_low_when_face_forward():
    """Mesmas mãos no colo, mas ROSTO P/ FRENTE (nariz na linha das orelhas) -> NÃO usa.

    É a porta do ângulo do rosto: mãos paradas no colo não bastam sem cabeça baixa.
    """
    score = phone_use_posture(_hidden_hands_skeleton(nose_drop=0.0), (90, 50, 210, 320), CFG)
    assert score <= 0.45, f"esperava baixo (rosto erguido), veio {score:.2f}"


def test_hidden_hands_sensitivity_scale():
    """A sensibilidade controla quanto de rosto baixo é exigido (precisão x recall)."""
    mild = _hidden_hands_skeleton(nose_drop=9.0)  # rosto levemente inclinado
    strict = Config(posture_hidden_sensitivity=0.0)   # exige rosto bem baixo
    loose = Config(posture_hidden_sensitivity=1.0)    # aceita pouca inclinação
    s_strict = phone_use_posture(mild, (90, 50, 210, 320), strict)
    s_loose = phone_use_posture(mild, (90, 50, 210, 320), loose)
    assert s_loose > s_strict, f"mais sensível deveria pontuar mais ({s_loose:.2f} vs {s_strict:.2f})"
    assert s_loose >= loose.posture_standalone_threshold
    assert s_strict < strict.posture_standalone_threshold


def test_hidden_hands_disabled_falls_back():
    """Com o sinal de mãos escondidas desligado, a postura escondida não marca."""
    cfg = Config(posture_hidden_enabled=False)
    score = phone_use_posture(_hidden_hands_skeleton(nose_drop=40.0), (90, 50, 210, 320), cfg)
    assert score < cfg.posture_standalone_threshold


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


# ===========================================================================
# Gate ROSTO → MÃOS (anti-falso-positivo) e POSTURA DE PERFIL (de lado)
# ===========================================================================
# Esqueletos de PERFIL calibrados sobre os vídeos reais data/view_side_*.mp4:
# de lado os ombros colapsam (sw≈0), só UMA orelha aparece e o nariz se desloca
# ~76px para o lado que o rosto aponta; quem usa olha para baixo (nariz ~37px
# abaixo da orelha) com as mãos no mesmo lado e bem abaixo do rosto.
PROFILE_BOX = (260.0, 60.0, 480.0, 530.0)  # w=220, h=470 -> sw/boxw pequeno


def _profile_using_skeleton() -> np.ndarray:
    """De lado USANDO: rosto virado p/ direita e baixo, mãos à direita e abaixo."""
    kp = np.zeros((17, 3), dtype=np.float32)
    # Ombros sobrepostos (perfil): quase no mesmo x.
    kp[KP_L_SHOULDER] = (300.0, 180.0, 0.9)
    kp[KP_R_SHOULDER] = (305.0, 180.0, 0.9)
    # Só a orelha direita (perfil); nariz à direita e abaixo dela.
    kp[KP_R_EAR] = (300.0, 100.0, 0.9)
    kp[KP_NOSE] = (376.0, 137.0, 0.9)
    # Mãos à direita (lado que o rosto aponta) e bem abaixo do rosto.
    kp[KP_L_WRIST] = (440.0, 360.0, 0.9)
    kp[KP_R_WRIST] = (450.0, 365.0, 0.9)
    # Cotovelo dobrado (antebraço segurando o aparelho).
    kp[KP_R_ELBOW] = (400.0, 300.0, 0.9)
    kp[KP_L_ELBOW] = (395.0, 295.0, 0.9)
    return kp


def _profile_wrong_side_skeleton() -> np.ndarray:
    """De lado: rosto virado p/ DIREITA mas mãos à ESQUERDA (não está olhando)."""
    kp = _profile_using_skeleton()
    # Pulsos do lado oposto ao que o rosto aponta.
    kp[KP_L_WRIST] = (250.0, 470.0, 0.9)
    kp[KP_R_WRIST] = (255.0, 470.0, 0.9)
    return kp


def _profile_head_up_skeleton() -> np.ndarray:
    """De lado: mãos no lado certo, mas cabeça ERGUIDA (olhando para frente)."""
    kp = _profile_using_skeleton()
    # Nariz ~na linha da orelha (rosto não está baixo).
    kp[KP_NOSE] = (376.0, 101.0, 0.9)
    return kp


def test_profile_view_detected():
    """Ombros colapsados (de lado) são reconhecidos como perfil; frontal não."""
    assert _is_profile_view(_profile_using_skeleton(), PROFILE_BOX, CFG.wrist_conf_threshold, CFG)
    assert not _is_profile_view(_using_skeleton(), (90, 50, 210, 300), CFG.wrist_conf_threshold, CFG)


def test_profile_using_flagged():
    """De lado USANDO (rosto baixo p/ as mãos) -> postura alta + marca uso."""
    score = phone_use_posture(_profile_using_skeleton(), PROFILE_BOX, CFG)
    assert score >= CFG.posture_standalone_threshold, f"esperava alto, veio {score:.2f}"
    person = PersonDetection(class_id=0, confidence=0.9, box=PROFILE_BOX,
                             keypoints=_profile_using_skeleton())
    out = Detector(config=CFG).associate([person], phones=[])
    assert out[0].using_phone is True and out[0].by_posture is True


def test_profile_wrong_side_not_flagged():
    """Rosto p/ um lado e mãos para o outro -> NÃO está olhando p/ elas -> 0."""
    score = phone_use_posture(_profile_wrong_side_skeleton(), PROFILE_BOX, CFG)
    assert score < CFG.posture_standalone_threshold, f"esperava baixo, veio {score:.2f}"
    person = PersonDetection(class_id=0, confidence=0.9, box=PROFILE_BOX,
                             keypoints=_profile_wrong_side_skeleton())
    out = Detector(config=CFG).associate([person], phones=[])
    assert out[0].using_phone is False


def test_profile_head_up_not_flagged():
    """De lado com a cabeça ERGUIDA (não olha p/ as mãos) -> não marca uso."""
    score = phone_use_posture(_profile_head_up_skeleton(), PROFILE_BOX, CFG)
    assert score < CFG.posture_standalone_threshold, f"esperava baixo, veio {score:.2f}"


def test_profile_view_when_no_shoulders():
    """Sem ombros confiáveis, tratamos como perfil (cai na via rosto→mãos)."""
    kp = _profile_using_skeleton()
    kp[KP_L_SHOULDER] = (0.0, 0.0, 0.0)  # ombros não confiáveis
    kp[KP_R_SHOULDER] = (0.0, 0.0, 0.0)
    assert _is_profile_view(kp, PROFILE_BOX, CFG.wrist_conf_threshold, CFG)


def test_face_toward_hands_high_when_looking_at_them():
    """Frontal olhando p/ baixo nas mãos -> alinhamento alto."""
    assert face_toward_hands(_using_skeleton(), CFG) >= CFG.face_hands_min_score


def test_face_toward_hands_zero_when_head_up():
    """Mãos no colo mas rosto p/ frente (cabeça erguida) -> alinhamento ~0."""
    kp = _hidden_hands_skeleton(nose_drop=0.0)
    assert face_toward_hands(kp, CFG) < CFG.face_hands_min_score


def test_face_gate_blocks_standalone_posture():
    """Postura forte mas rosto NÃO virado p/ as mãos -> gate bloqueia o uso autônomo.

    Constrói uma pose com score de postura alto mas mãos do lado oposto ao rosto;
    o gate (face_to_hands baixo) impede marcar uso sem ver o aparelho.
    """
    person = PersonDetection(class_id=0, confidence=0.9, box=PROFILE_BOX,
                             keypoints=_profile_wrong_side_skeleton())
    out = Detector(config=CFG).associate([person], phones=[])
    assert out[0].face_to_hands < CFG.face_hands_min_score
    assert out[0].using_phone is False


def test_face_gate_disabled_allows_posture():
    """Com o gate desligado, a postura volta a decidir sozinha (sem exigir rosto)."""
    cfg = Config(posture_face_gate_enabled=False, posture_profile_enabled=False)
    # Frontal "usando" continua marcando independentemente do gate.
    person = PersonDetection(class_id=0, confidence=0.9, box=(90, 50, 210, 300),
                             keypoints=_using_skeleton())
    out = Detector(config=cfg).associate([person], phones=[])
    assert out[0].using_phone is True


def test_visible_phone_overrides_face_gate():
    """Celular REALMENTE detectado na mão marca uso mesmo sem o rosto p/ as mãos.

    O gate rosto→mãos só restringe o sinal AUTÔNOMO (sem caixa). Vendo o aparelho
    no pulso, a associação por proximidade decide — 'só com o celular visível
    gera positivo', como pedido.
    """
    # Pessoa de lado com a cabeça erguida (gate de postura falharia)...
    kp = _profile_head_up_skeleton()
    person = PersonDetection(class_id=0, confidence=0.9, box=PROFILE_BOX, keypoints=kp)
    # ...mas há um celular detectado exatamente sobre o pulso direito.
    phone = Detection(class_id=CFG.phone_class_id, confidence=0.6,
                      box=(440.0, 355.0, 470.0, 380.0))
    out = Detector(config=CFG).associate([person], phones=[phone])
    assert out[0].using_phone is True
    assert out[0].by_posture is False  # veio do celular visível, não da postura
