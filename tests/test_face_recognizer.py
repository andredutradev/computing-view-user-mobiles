"""Testes do reconhecedor facial (InsightFace mockado) e do cache de identidade."""

from __future__ import annotations

import numpy as np

from src.attendance.enrollment import Gallery, StudentRecord
from src.attendance.face_recognizer import FaceRecognizer, IdentityCache
from src.attendance.geometry import normalize
from src.config import Config


class _FakeFace:
    def __init__(self, emb, bbox, score=0.9):
        self.normed_embedding = np.asarray(emb, dtype=np.float32)
        self.bbox = bbox
        self.det_score = score


class _FakeApp:
    """Imita insightface FaceAnalysis: .get(img) -> lista de rostos."""

    def __init__(self, faces):
        self._faces = faces

    def get(self, image):
        return self._faces


def _gallery():
    return Gallery(
        {
            "ana": StudentRecord(
                "ana", "Ana", np.array([normalize([1, 0, 0])], dtype=np.float32), (), ""
            ),
            "bob": StudentRecord(
                "bob", "Bob", np.array([normalize([0, 1, 0])], dtype=np.float32), (), ""
            ),
        }
    )


def test_embed_full_frame_wraps_faces():
    faces = [
        _FakeFace([1, 0, 0], (0, 0, 10, 10)),
        _FakeFace([0, 1, 0], (0, 0, 40, 40)),
    ]
    rec = FaceRecognizer(config=Config(), app=_FakeApp(faces))
    out = rec.embed_full_frame(np.zeros((50, 50, 3), dtype=np.uint8))
    assert len(out) == 2
    # embed_face devolve o MAIOR rosto (área 40x40).
    largest = rec.embed_face(np.zeros((50, 50, 3), dtype=np.uint8))
    assert np.allclose(largest, [0, 1, 0])


def test_match_returns_best_above_threshold():
    rec = FaceRecognizer(config=Config(), app=object())
    sid, score = rec.match(np.array([1, 0, 0], dtype=np.float32), _gallery(), 0.35)
    assert sid == "ana"
    assert score > 0.9


def test_match_returns_none_below_threshold():
    rec = FaceRecognizer(config=Config(), app=object())
    # Vetor ortogonal a todos -> similaridade ~0 < limiar.
    sid, score = rec.match(np.array([0, 0, 1], dtype=np.float32), _gallery(), 0.35)
    assert sid is None


def test_identity_cache_cadence_and_confirmation():
    cache = IdentityCache()
    cfg = Config(
        face_recog_every_n_frames=5,
        face_confirm_hits=2,
        face_confirm_score=0.99,
        face_match_threshold=0.35,
    )
    assert cache.should_recognize(1, 0, cfg) is True
    cache.update(1, "ana", 0.6, 0, cfg)  # hit 1 (não confirma: 0.6<0.99, hits<2)
    assert cache.identity_of(1) == "ana"
    assert cache.is_confirmed(1) is False
    assert cache.should_recognize(1, 1, cfg) is False  # gap 1 < 5
    assert cache.should_recognize(1, 5, cfg) is True  # gap 5 >= 5
    cache.update(1, "ana", 0.6, 5, cfg)  # hit 2 -> confirma
    assert cache.is_confirmed(1) is True
    assert cache.should_recognize(1, 100, cfg) is False  # confirmado -> nunca mais


def test_identity_cache_confirms_on_high_score():
    cache = IdentityCache()
    cfg = Config(face_confirm_score=0.5, face_confirm_hits=10, face_match_threshold=0.35)
    cache.update(7, "bob", 0.8, 0, cfg)  # score alto -> confirma de imediato
    assert cache.is_confirmed(7) is True


def test_identity_cache_gives_up_after_max_attempts():
    cache = IdentityCache()
    cfg = Config(face_max_attempts=2, face_recog_every_n_frames=1, face_match_threshold=0.9)
    cache.update(3, None, 0.1, 0, cfg)  # tentativa 1 (sem match)
    cache.update(3, None, 0.1, 1, cfg)  # tentativa 2
    assert cache.should_recognize(3, 2, cfg) is False  # desistiu
    assert cache.identity_of(3) is None
