"""Testes do harness de validação por imagens (``src/eval_images.py``).

Estratégia: não usar YOLO/disco reais. Geramos imagens de verdade num diretório
temporário (com ``cv2.imwrite``) e injetamos um ``Detector`` com modelos fake que
devolvem uma cena controlada (uma pessoa segurando um celular), de modo que a
regra pulso↔celular marque ``using_phone=True``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.detector import (
    KP_L_SHOULDER,
    KP_R_SHOULDER,
    KP_R_WRIST,
    Detector,
)
from src.eval_images import ImageResult, evaluate_folder, format_report, list_images

cv2 = pytest.importorskip("cv2")


# --- mocks no mesmo formato Results/boxes/keypoints do ultralytics ----------
class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)


class _FakeKeypoints:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeYOLO:
    def __init__(self, boxes, keypoints=None):
        self._boxes = boxes
        self._keypoints = keypoints

    def predict(self, frame, **kwargs):
        return [_FakeResult(self._boxes, self._keypoints)]


def _scene_detector() -> Detector:
    """Detector com uma pessoa (pulso direito sobre o celular) -> usando."""
    # Pessoa (100,50)-(200,400), ombros a 80px -> raio da mão = 0.5*80 = 40px.
    person_box = (100.0, 50.0, 200.0, 400.0)
    phone_box = (150.0, 250.0, 180.0, 300.0)
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[KP_L_SHOULDER] = (110.0, 120.0, 0.95)
    kp[KP_R_SHOULDER] = (190.0, 120.0, 0.95)
    kp[KP_R_WRIST] = (165.0, 275.0, 0.9)  # pulso no centro do celular

    phone_model = _FakeYOLO(_FakeBoxes([phone_box], [67], [0.4]))
    pose_model = _FakeYOLO(
        _FakeBoxes([person_box], [0], [0.9]), _FakeKeypoints([kp])
    )
    return Detector(config=Config(), model=phone_model, pose_model=pose_model)


def _write_image(path, color=0) -> None:
    img = np.full((480, 640, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_list_images_filters_and_sorts(tmp_path):
    _write_image(tmp_path / "b.jpg")
    _write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("ignore me")
    names = [p.name for p in list_images(tmp_path)]
    assert names == ["a.png", "b.jpg"]


def test_evaluate_folder_marks_using_phone(tmp_path):
    _write_image(tmp_path / "foto1.jpg")
    _write_image(tmp_path / "foto2.jpg")

    results = evaluate_folder(tmp_path, config=Config(), detector=_scene_detector())

    assert len(results) == 2
    assert all(r.using_phone for r in results)
    assert all(r.ok for r in results)
    assert all(r.people == 1 and r.phones == 1 for r in results)
    assert results[0].best_phone_conf == pytest.approx(0.4)


def test_evaluate_folder_unreadable_image_is_error(tmp_path):
    # Arquivo com extensão de imagem mas conteúdo inválido -> imread devolve None.
    bad = tmp_path / "corrompida.jpg"
    bad.write_bytes(b"not an image")
    results = evaluate_folder(tmp_path, config=Config(), detector=_scene_detector())
    assert len(results) == 1
    assert results[0].error is not None
    assert results[0].ok is False


def test_evaluate_folder_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_folder(tmp_path / "nao_existe", config=Config())


def test_format_report_has_rate_line():
    results = [
        ImageResult(path=type("P", (), {"name": "x.jpg"})(), people=1, phones=1, using_phone=True, best_phone_conf=0.5),
    ]
    text = format_report(results)
    assert "Acertos" in text
    assert "1/1" in text
