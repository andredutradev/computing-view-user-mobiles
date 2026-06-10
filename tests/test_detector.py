"""Testes unitários do Detector e da geometria.

Estratégia: NÃO baixar/usar o YOLO real. Em vez disso, injetamos modelos
*fake* (mocks) que devolvem detecções/keypoints controlados. Assim os testes
são rápidos, determinísticos e não dependem de rede nem de GPU.

São dois modelos no pipeline de pose:
  - modelo de DETECÇÃO -> caixas de celular (classe 67);
  - modelo de POSE     -> caixas de pessoa (classe 0) + 17 keypoints COCO.

Os frames são simplesmente arrays numpy (como os que o OpenCV produz).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.config import Config
from src.detector import (
    KP_L_SHOULDER,
    KP_L_WRIST,
    KP_R_SHOULDER,
    KP_R_WRIST,
    Detection,
    Detector,
    PersonDetection,
    containment_ratio,
    euclidean_distance,
    has_confident_wrist,
    intersection_area,
    iou,
    person_scale,
    point_box_distance,
    resolve_device,
    wrist_phone_proximity,
)


# ---------------------------------------------------------------------------
# Mocks que imitam a estrutura Results/boxes/keypoints do ultralytics
# ---------------------------------------------------------------------------
class _FakeBoxes:
    """Imita ``result.boxes`` expondo xyxy, cls e conf como numpy arrays."""

    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)


class _FakeKeypoints:
    """Imita ``result.keypoints`` expondo ``.data`` (N, 17, 3)."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeYOLO:
    """Modelo fake: devolve sempre o mesmo conjunto de detecções."""

    def __init__(self, boxes: _FakeBoxes, keypoints: _FakeKeypoints | None = None):
        self._boxes = boxes
        self._keypoints = keypoints
        self.predict_calls = 0

    def predict(self, frame, **kwargs):
        self.predict_calls += 1
        return [_FakeResult(self._boxes, self._keypoints)]


@pytest.fixture
def frame() -> np.ndarray:
    """Frame mockado de 480x640 (como um quadro de webcam)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# Pessoa de referência usada em vários testes: caixa (100,50)-(200,400),
# ombros a ~80px de distância (escala ~80 -> raio da mão = 0.5*80 = 40px).
PERSON_BOX = (100.0, 50.0, 200.0, 400.0)


def _person_kp(wrist_xy, wrist_conf=0.9) -> np.ndarray:
    """Keypoints (17,3) com ombros e o pulso DIREITO numa posição dada."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[KP_L_SHOULDER] = (110.0, 120.0, 0.95)
    kp[KP_R_SHOULDER] = (190.0, 120.0, 0.95)  # largura de ombros = 80px
    kp[KP_R_WRIST] = (wrist_xy[0], wrist_xy[1], wrist_conf)
    return kp


# ---------------------------------------------------------------------------
# Testes das funções geométricas puras (inalteradas)
# ---------------------------------------------------------------------------
def test_intersection_area_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert intersection_area(a, b) == 25.0


def test_intersection_area_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert intersection_area(a, b) == 0.0


def test_iou_identical_boxes():
    a = (0, 0, 10, 10)
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_half_overlap():
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    assert iou(a, b) == pytest.approx(1 / 3)


def test_containment_phone_inside_person():
    person = (0, 0, 100, 200)
    phone = (40, 80, 60, 120)
    assert containment_ratio(inner=phone, outer=person) == pytest.approx(1.0)


def test_euclidean_distance_centers():
    a = (0, 0, 10, 10)
    b = (0, 0, 10, 30)
    assert euclidean_distance(a, b) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Testes das novas funções de pulso/mão
# ---------------------------------------------------------------------------
def test_point_box_distance_inside_is_zero():
    assert point_box_distance(15, 15, (10, 10, 20, 20)) == 0.0


def test_point_box_distance_outside():
    # Ponto (10, 13) à esquerda da caixa [20..30] em x -> dx=10, dy=0.
    assert point_box_distance(10, 13, (20, 10, 30, 20)) == pytest.approx(10.0)


def test_person_scale_uses_shoulder_width():
    cfg = Config()
    kp = _person_kp((165, 175))
    # Ombros em x=110 e x=190 -> largura 80.
    assert person_scale(kp, PERSON_BOX, cfg) == pytest.approx(80.0)


def test_person_scale_fallback_to_box_width():
    cfg = Config()
    # Sem keypoints confiáveis de ombro -> usa largura da caixa (100).
    assert person_scale(None, PERSON_BOX, cfg) == pytest.approx(100.0)


def test_wrist_proximity_holding_when_wrist_on_phone():
    cfg = Config()
    kp = _person_kp((165, 175))  # pulso dentro do celular
    phone = (150.0, 150.0, 180.0, 200.0)
    holding, wrist, score = wrist_phone_proximity(kp, phone, PERSON_BOX, cfg)
    assert holding is True
    assert wrist == pytest.approx((165.0, 175.0))
    assert score == 0.0


def test_wrist_proximity_not_holding_when_far():
    cfg = Config()
    kp = _person_kp((165, 360))  # pulso bem abaixo do celular
    phone = (150.0, 150.0, 180.0, 200.0)
    holding, _, _ = wrist_phone_proximity(kp, phone, PERSON_BOX, cfg)
    assert holding is False


def test_wrist_proximity_ignores_low_confidence_wrist():
    cfg = Config()
    kp = _person_kp((165, 175), wrist_conf=0.1)  # pulso sobre o celular, mas incerto
    phone = (150.0, 150.0, 180.0, 200.0)
    holding, _, _ = wrist_phone_proximity(kp, phone, PERSON_BOX, cfg)
    assert holding is False
    assert has_confident_wrist(kp, cfg) is False


# ---------------------------------------------------------------------------
# Testes da classe Detector
# ---------------------------------------------------------------------------
def test_detector_initializes_without_real_model():
    """A classe deve instanciar sem baixar o YOLO (modelos injetados)."""
    detector = Detector(model=object(), pose_model=object())
    assert detector is not None
    assert detector.config is not None


def test_detect_phones_parses_only_phones(frame):
    boxes = _FakeBoxes(xyxy=[[150, 150, 180, 200]], cls=[67], conf=[0.8])
    detector = Detector(model=_FakeYOLO(boxes))
    phones = detector.detect_phones(frame)
    assert len(phones) == 1
    assert phones[0].class_id == 67
    assert all(isinstance(p, Detection) for p in phones)


def test_detect_people_parses_boxes_and_keypoints(frame):
    boxes = _FakeBoxes(xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9])
    kp_data = _person_kp((165, 175))[None, :, :]  # (1, 17, 3)
    pose = _FakeYOLO(boxes, _FakeKeypoints(kp_data))
    detector = Detector(pose_model=pose)
    people = detector.detect_people(frame)
    assert len(people) == 1
    assert isinstance(people[0], PersonDetection)
    assert people[0].keypoints is not None
    assert people[0].keypoints.shape == (17, 3)


def test_process_frame_marks_using_when_wrist_on_phone(frame):
    """Pulso sobre o celular -> 'usando' (sinal primário de pose)."""
    phone_boxes = _FakeBoxes(xyxy=[[150, 150, 180, 200]], cls=[67], conf=[0.8])
    person_boxes = _FakeBoxes(xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9])
    kp = _person_kp((165, 175))[None, :, :]
    detector = Detector(
        model=_FakeYOLO(phone_boxes),
        pose_model=_FakeYOLO(person_boxes, _FakeKeypoints(kp)),
    )
    people = detector.process_frame(frame)
    assert len(people) == 1
    assert people[0].using_phone is True
    assert people[0].matched_phone is not None
    assert people[0].holding_wrist == pytest.approx((165.0, 175.0))


def test_process_frame_not_using_when_phone_in_body_but_far_from_hand(frame):
    """Ganho de precisão: celular dentro do corpo, mas longe da MÃO -> NÃO usa."""
    # Celular contido na caixa da pessoa (contenção alta), porém o pulso visível
    # está bem longe dele. O comportamento antigo marcaria verde; o novo não.
    phone_boxes = _FakeBoxes(xyxy=[[150, 150, 180, 200]], cls=[67], conf=[0.8])
    person_boxes = _FakeBoxes(xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9])
    kp = _person_kp((165, 360))[None, :, :]  # pulso confiável, mas distante
    detector = Detector(
        model=_FakeYOLO(phone_boxes),
        pose_model=_FakeYOLO(person_boxes, _FakeKeypoints(kp)),
    )
    people = detector.process_frame(frame)
    assert people[0].using_phone is False
    assert people[0].matched_phone is None


def test_process_frame_fallback_containment_without_keypoints(frame):
    """Sem keypoints (pulsos não visíveis) -> cai no fallback de contenção."""
    phone_boxes = _FakeBoxes(xyxy=[[140, 150, 175, 210]], cls=[67], conf=[0.8])
    person_boxes = _FakeBoxes(xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9])
    # Pose sem keypoints (result.keypoints = None) -> person.keypoints = None.
    detector = Detector(
        model=_FakeYOLO(phone_boxes),
        pose_model=_FakeYOLO(person_boxes, keypoints=None),
    )
    people = detector.process_frame(frame)
    assert people[0].keypoints is None
    assert people[0].using_phone is True  # contenção marca (celular dentro)
    assert people[0].matched_phone is not None


def test_process_frame_picks_person_with_closest_wrist(frame):
    """Com duas pessoas, o celular vai para a do PULSO mais próximo."""
    phone_boxes = _FakeBoxes(xyxy=[[150, 150, 180, 200]], cls=[67], conf=[0.8])
    person_boxes = _FakeBoxes(
        xyxy=[[100, 50, 200, 400], [400, 50, 500, 400]],
        cls=[0, 0],
        conf=[0.9, 0.9],
    )
    kp_a = _person_kp((165, 175))  # pulso sobre o celular
    kp_b = np.zeros((17, 3), dtype=np.float32)
    kp_b[KP_L_SHOULDER] = (410, 120, 0.95)
    kp_b[KP_R_SHOULDER] = (490, 120, 0.95)
    kp_b[KP_R_WRIST] = (460, 175, 0.9)  # longe do celular
    kp = np.stack([kp_a, kp_b])  # (2, 17, 3)
    detector = Detector(
        model=_FakeYOLO(phone_boxes),
        pose_model=_FakeYOLO(person_boxes, _FakeKeypoints(kp)),
    )
    people = detector.process_frame(frame)
    using = [p for p in people if p.using_phone]
    assert len(using) == 1
    assert using[0].box[0] == 100  # pessoa A (pulso próximo)


def test_process_frame_pose_disabled_uses_containment(frame):
    """Com pose DESLIGADA, um único modelo detecta tudo e usa contenção."""
    cfg = Config(pose_enabled=False)
    boxes = _FakeBoxes(
        xyxy=[[100, 50, 200, 400], [140, 150, 175, 210]],
        cls=[0, 67],
        conf=[0.9, 0.8],
    )
    detector = Detector(config=cfg, model=_FakeYOLO(boxes))
    people = detector.process_frame(frame)
    assert len(people) == 1
    assert people[0].keypoints is None
    assert people[0].using_phone is True


def test_process_frame_rejects_invalid_frame():
    """Frame inválido (None) deve levantar ValueError."""
    detector = Detector(model=object(), pose_model=object())
    with pytest.raises(ValueError):
        detector.process_frame(None)  # type: ignore[arg-type]


def test_custom_config_thresholds_block_far_wrist(frame):
    """Raio menor (config injetada) impede match de pulso um pouco distante."""
    # Pulso a ~50px da borda do celular; raio padrão (40) já bloquearia, então
    # usamos um pulso a 20px e raio reduzido para 0.1*80 = 8px.
    strict = Config(hand_radius_factor=0.1)
    phone_boxes = _FakeBoxes(xyxy=[[150, 150, 180, 200]], cls=[67], conf=[0.8])
    person_boxes = _FakeBoxes(xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9])
    kp = _person_kp((200, 175))[None, :, :]  # 20px à direita da borda (x=180)
    detector = Detector(
        config=strict,
        model=_FakeYOLO(phone_boxes),
        pose_model=_FakeYOLO(person_boxes, _FakeKeypoints(kp)),
    )
    people = detector.process_frame(frame)
    assert people[0].using_phone is False


# ---------------------------------------------------------------------------
# Testes de seleção de dispositivo (adaptativa)
# ---------------------------------------------------------------------------
def _fake_torch(cuda: bool, mps: bool) -> types.ModuleType:
    """Monta um módulo torch fake com disponibilidade controlada de cuda/mps."""
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    fake.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return fake


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"
    assert resolve_device("0") == "0"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=True))
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    assert resolve_device("auto") == "cpu"


# ---------------------------------------------------------------------------
# Testes de rastreamento (track_id)
# ---------------------------------------------------------------------------
class _FakeBoxesTracked(_FakeBoxes):
    """Como _FakeBoxes, mas também expõe ``id`` (track IDs), como em track()."""

    def __init__(self, xyxy, cls, conf, ids):
        super().__init__(xyxy, cls, conf)
        self.id = np.asarray(ids, dtype=np.float32)


class _FakeYOLOTrack:
    """Modelo fake com .track() (em vez de .predict()), devolvendo IDs."""

    def __init__(self, boxes, keypoints=None):
        self._boxes = boxes
        self._keypoints = keypoints
        self.track_calls = 0

    def track(self, frame, **kwargs):
        self.track_calls += 1
        return [_FakeResult(self._boxes, self._keypoints)]


def test_detect_people_tracked_reads_track_id(frame):
    boxes = _FakeBoxesTracked(
        xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9], ids=[7]
    )
    kp = _person_kp((165, 175))[None, :, :]
    pose = _FakeYOLOTrack(boxes, _FakeKeypoints(kp))
    detector = Detector(pose_model=pose)
    people = detector.detect_people_tracked(frame)
    assert len(people) == 1
    assert people[0].track_id == 7
    assert pose.track_calls == 1


def test_detect_people_without_id_has_none_track(frame):
    """Caminho predict puro (sem .id) -> track_id None, sem quebrar."""
    boxes = _FakeBoxes(xyxy=[[100, 50, 200, 400]], cls=[0], conf=[0.9])
    kp = _person_kp((165, 175))[None, :, :]
    pose = _FakeYOLO(boxes, _FakeKeypoints(kp))
    detector = Detector(pose_model=pose)
    people = detector.detect_people(frame)
    assert people[0].track_id is None
