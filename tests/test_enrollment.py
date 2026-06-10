"""Testes de matrícula e galeria (sem InsightFace — recognizer mockado)."""

from __future__ import annotations

import numpy as np

from src.attendance.enrollment import (
    Gallery,
    StudentRecord,
    enroll_capture,
    enroll_directory,
    slugify,
)
from src.attendance.face_recognizer import Face
from src.config import Config

# Augmentation desligada nestes testes para isolar a contagem foto→embedding;
# o fake devolve embeddings DISTINTOS por chamada (como fotos reais).
_NO_AUG = Config(face_enroll_augment=False)


class _FakeRecognizer:
    """Recognizer fake: devolve embeddings DISTINTOS a cada imagem lida.

    Fotos reais geram vetores diferentes; o contador garante isso para que a
    deduplicação por cosseno não colapse fotos legítimas (e ainda colapse
    duplicatas verdadeiras quando o mesmo vetor reaparece).
    """

    def __init__(self, faces_per_image=1):
        self.faces_per_image = faces_per_image
        self._n = 0

    def embed_full_frame(self, image):
        out = []
        for _ in range(self.faces_per_image):
            self._n += 1
            emb = np.zeros(8, dtype=np.float32)
            emb[self._n % 8] = 1.0  # vetor unitário distinto por chamada
            out.append(Face(embedding=emb, bbox=(0.0, 0.0, 50.0, 50.0), det_score=0.9))
        return out


def test_slugify():
    assert slugify("Ana Silva") == "ana_silva"
    assert slugify("  Bruno-Costa ") == "bruno_costa"
    # Acentos são removidos para o id (mas preservados no display_name).
    assert slugify("André Conceição") == "andre_conceicao"


def test_gallery_round_trip(tmp_path):
    rec = StudentRecord(
        "ana",
        "Ana Silva",
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        ("a.jpg",),
        "2026-06-07",
    )
    gallery = Gallery({"ana": rec})
    path = tmp_path / "gallery.npz"
    gallery.save(path)
    assert (tmp_path / "gallery.json").exists()

    loaded = Gallery.load(path)
    assert "ana" in loaded.records
    assert loaded.records["ana"].display_name == "Ana Silva"
    assert loaded.records["ana"].embeddings.shape == (2, 3)


def test_all_embeddings_parallel_ids():
    gallery = Gallery(
        {
            "a": StudentRecord("a", "A", np.eye(3, dtype=np.float32), (), ""),
            "b": StudentRecord("b", "B", np.ones((1, 3), dtype=np.float32), (), ""),
        }
    )
    matrix, ids = gallery.all_embeddings()
    assert matrix.shape == (4, 3)
    assert ids.count("a") == 3
    assert ids.count("b") == 1


def test_load_missing_returns_empty(tmp_path):
    gallery = Gallery.load(tmp_path / "nao_existe.npz")
    assert len(gallery) == 0


def test_remove_student():
    gallery = Gallery(
        {"a": StudentRecord("a", "A", np.eye(3, dtype=np.float32), (), "")}
    )
    gallery.remove("a")
    assert len(gallery) == 0


def test_enroll_directory_with_subfolder_and_flat(tmp_path, monkeypatch):
    import cv2

    # Estrutura: subpasta (ana_silva) + imagem plana (bruno).
    (tmp_path / "ana_silva").mkdir()
    (tmp_path / "ana_silva" / "f1.jpg").write_bytes(b"x")
    (tmp_path / "ana_silva" / "f2.jpg").write_bytes(b"x")
    (tmp_path / "bruno.png").write_bytes(b"x")

    # cv2.imread mockado: o conteúdo do arquivo é irrelevante para o teste.
    monkeypatch.setattr(
        cv2, "imread", lambda p: np.zeros((80, 80, 3), dtype=np.uint8)
    )

    gallery, warnings = enroll_directory(
        tmp_path, _FakeRecognizer(), gallery_path=tmp_path / "g.npz", config=_NO_AUG
    )
    assert "ana_silva" in gallery.records
    assert "bruno" in gallery.records
    # Ana tem 2 fotos -> 2 embeddings (distintos); Bruno, 1.
    assert gallery.records["ana_silva"].embeddings.shape[0] == 2
    assert gallery.records["bruno"].embeddings.shape[0] == 1


def test_enroll_directory_augmentation_adds_embeddings(tmp_path, monkeypatch):
    """Com augmentation ligada, UMA foto rende mais de um embedding."""
    import cv2

    (tmp_path / "carla.jpg").write_bytes(b"x")
    monkeypatch.setattr(
        cv2, "imread", lambda p: np.zeros((80, 80, 3), dtype=np.uint8)
    )
    gallery, _ = enroll_directory(
        tmp_path, _FakeRecognizer(), gallery_path=None,
        config=Config(face_enroll_augment=True),
    )
    # original + espelho = 2 embeddings distintos de uma única foto.
    assert gallery.records["carla"].embeddings.shape[0] == 2


def test_enroll_directory_quality_gate_rejects_low_score(tmp_path, monkeypatch):
    """Rosto com det_score abaixo do limiar é ignorado (gate de qualidade)."""
    import cv2

    class _LowScoreRec:
        def embed_full_frame(self, image):
            return [Face(embedding=np.ones(8, dtype=np.float32),
                         bbox=(0.0, 0.0, 50.0, 50.0), det_score=0.2)]

    (tmp_path / "dudu.jpg").write_bytes(b"x")
    monkeypatch.setattr(cv2, "imread", lambda p: np.zeros((80, 80, 3), dtype=np.uint8))
    gallery, warnings = enroll_directory(
        tmp_path, _LowScoreRec(), gallery_path=None,
        config=Config(face_enroll_min_det_score=0.5),
    )
    assert "dudu" not in gallery.records
    assert any("baixa qualidade" in w for w in warnings)


def test_enroll_capture_saves_photo_and_gallery(tmp_path):
    """Captura única: grava a foto, cria a galeria e devolve ok=True."""
    cfg = Config(
        students_dir=str(tmp_path / "students"),
        gallery_path=str(tmp_path / "students" / "gallery.npz"),
        face_enroll_augment=False,
    )
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    ok, msg, gallery = enroll_capture(
        "André Conceição", image, _FakeRecognizer(), cfg=cfg
    )
    assert ok is True
    assert "andre_conceicao" in gallery.records
    assert gallery.records["andre_conceicao"].display_name == "André Conceição"
    # Foto de referência persistida em data/students/<slug>/.
    saved = list((tmp_path / "students" / "andre_conceicao").glob("*.jpg"))
    assert len(saved) == 1
    # A galeria foi salva em disco.
    assert (tmp_path / "students" / "gallery.npz").exists()


def test_enroll_capture_requires_name():
    ok, msg, _ = enroll_capture("", np.zeros((10, 10, 3), np.uint8), _FakeRecognizer())
    assert ok is False
    assert "nome" in msg.lower()


def test_enroll_capture_no_face(tmp_path):
    cfg = Config(
        students_dir=str(tmp_path / "students"),
        gallery_path=str(tmp_path / "g.npz"),
    )
    ok, msg, _ = enroll_capture(
        "Ana", np.zeros((80, 80, 3), np.uint8), _FakeRecognizer(faces_per_image=0), cfg=cfg
    )
    assert ok is False
    assert "rosto" in msg.lower()


def test_enroll_directory_warns_on_no_faces(tmp_path, monkeypatch):
    import cv2

    (tmp_path / "carla.jpg").write_bytes(b"x")
    monkeypatch.setattr(
        cv2, "imread", lambda p: np.zeros((80, 80, 3), dtype=np.uint8)
    )
    gallery, warnings = enroll_directory(
        tmp_path, _FakeRecognizer(faces_per_image=0), gallery_path=None
    )
    assert "carla" not in gallery.records
    assert any("0 rostos" in w for w in warnings)
