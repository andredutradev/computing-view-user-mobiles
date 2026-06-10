"""Reconhecimento facial com InsightFace + cache de identidade por track.

``FaceRecognizer`` encapsula o ``insightface.app.FaceAnalysis`` (pacote
``buffalo_l``), com seleção ADAPTATIVA de provider (CUDA se disponível, senão
CPU). O carregamento é *lazy* (espelha ``Detector.load``) e o ``app`` é
INJETÁVEL — em testes passa-se um mock e nada de InsightFace é importado.

``IdentityCache`` amarra ``track_id -> aluno``: o reconhecimento (caro) só roda
em tracks ainda NÃO confirmados e a cada N frames, sendo então cacheado. É o que
mantém o tempo real em CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attendance.geometry import normalize
from src.config import Config, settings


@dataclass
class Face:
    """Um rosto detectado: embedding normalizado + caixa + score."""

    embedding: np.ndarray  # (D,) L2-normalizado
    bbox: tuple  # (x1, y1, x2, y2)
    det_score: float = 0.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class FaceRecognizer:
    """Wrapper do InsightFace com providers adaptativos (CPU/GPU)."""

    def __init__(self, config: Config | None = None, app: object | None = None) -> None:
        self.config = config or settings
        self._app = app  # injete um mock em testes; None -> carrega o real

    # -- ciclo de vida ------------------------------------------------------
    def load(self) -> None:
        """Carrega o FaceAnalysis do InsightFace (uma vez)."""
        if self._app is not None:
            return
        providers = self._providers()
        # Import LOCAL: importar este módulo não exige insightface instalado.
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=self.config.face_model_pack, providers=providers)
        ctx_id = 0 if any("CUDA" in p for p in providers) else -1
        app.prepare(
            ctx_id=ctx_id,
            det_size=(self.config.face_det_size, self.config.face_det_size),
        )
        self._app = app

    @property
    def app(self) -> object:
        if self._app is None:
            self.load()
        return self._app

    def _providers(self) -> list:
        """Escolhe os execution providers do onnxruntime (CUDA -> CPU)."""
        available: list = []
        try:
            import onnxruntime as ort

            available = list(ort.get_available_providers())
        except Exception:  # pragma: no cover - onnxruntime ausente
            available = []
        if (
            self.config.face_use_gpu_if_available
            and "CUDAExecutionProvider" in available
        ):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # -- inferência ---------------------------------------------------------
    def embed_full_frame(self, image: np.ndarray) -> list:
        """Detecta TODOS os rostos em ``image`` e devolve uma lista de ``Face``."""
        if image is None or getattr(image, "size", 0) == 0:
            return []
        faces = self.app.get(image)
        out: list = []
        for f in faces or []:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = normalize(getattr(f, "embedding"))
            else:
                emb = np.asarray(emb, dtype=np.float32)
            bbox = tuple(float(v) for v in getattr(f, "bbox", (0.0, 0.0, 0.0, 0.0)))
            out.append(
                Face(embedding=emb, bbox=bbox, det_score=float(getattr(f, "det_score", 0.0)))
            )
        return out

    def embed_face(self, crop: np.ndarray) -> np.ndarray | None:
        """Embedding do MAIOR rosto em ``crop`` (já recortado), ou None.

        Com ``face_tta_flip`` ligado, faz **Test-Time Augmentation**: embute o
        rosto e o seu espelho horizontal e devolve a MÉDIA (normalizada) dos
        dois — mais robusto a leves variações de pose/iluminação.
        """
        emb = self._largest_embedding(crop)
        if emb is None:
            return None
        if self.config.face_tta_flip:
            import cv2

            flipped = self._largest_embedding(cv2.flip(crop, 1))
            if flipped is not None:
                emb = normalize(np.asarray(emb, dtype=np.float32) + flipped)
        return emb

    def _largest_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Embedding (normalizado) do maior rosto em ``crop``, ou None."""
        faces = self.embed_full_frame(crop)
        if not faces:
            return None
        return normalize(max(faces, key=lambda f: f.area).embedding)

    def match(self, embedding: np.ndarray, gallery, threshold: float, margin: float | None = None):
        """Correspondência por cosseno do ``embedding`` contra a galeria.

        Agrega por ALUNO (melhor similaridade entre os embeddings do aluno),
        evitando que um aluno com muitas fotos domine só pela quantidade. Aplica
        uma **margem** anti-ambiguidade: se o melhor e o 2º melhor aluno estão
        muito próximos (diferença < ``margin``), recusa (devolve ``None``).

        Devolve ``(student_id, score)`` se aprovado, senão ``(None, score)``.
        """
        matrix, ids = gallery.all_embeddings()
        if matrix is None or len(matrix) == 0 or not ids:
            return (None, 0.0)
        if margin is None:
            margin = self.config.face_match_margin
        emb = normalize(embedding)
        sims = matrix @ emb

        # Melhor similaridade por aluno (não por linha/foto).
        best_by_sid: dict = {}
        for sid, s in zip(ids, sims):
            s = float(s)
            if s > best_by_sid.get(sid, -1.0):
                best_by_sid[sid] = s
        ranked = sorted(best_by_sid.items(), key=lambda kv: kv[1], reverse=True)
        best_sid, best_score = ranked[0]

        if best_score < threshold:
            return (None, best_score)
        # Anti-ambiguidade: exige folga sobre o 2º melhor aluno.
        if margin > 0.0 and len(ranked) > 1 and (best_score - ranked[1][1]) < margin:
            return (None, best_score)
        return (best_sid, best_score)


# ---------------------------------------------------------------------------
# Cache de identidade por track_id
# ---------------------------------------------------------------------------
@dataclass
class TrackIdentity:
    track_id: int
    student_id: str | None = None
    score: float = 0.0
    hits: int = 0
    attempts: int = 0
    confirmed: bool = False
    last_attempt_frame: int = -10**9


class IdentityCache:
    """Decide QUANDO reconhecer e memoriza a identidade de cada track."""

    def __init__(self) -> None:
        self._by_track: dict = {}  # dict[int, TrackIdentity]

    def should_recognize(self, track_id: int, frame_index: int, cfg: Config) -> bool:
        """True se vale a pena (re)tentar reconhecer este track agora."""
        ti = self._by_track.get(track_id)
        if ti is None:
            return True  # nunca tentamos -> tenta já
        if ti.confirmed:
            return False  # já confirmado -> nunca mais
        if ti.attempts >= cfg.face_max_attempts:
            return False  # desistimos (fica "?")
        return (frame_index - ti.last_attempt_frame) >= cfg.face_recog_every_n_frames

    def update(
        self,
        track_id: int,
        student_id: str | None,
        score: float,
        frame_index: int,
        cfg: Config,
    ) -> None:
        """Incorpora o resultado de uma tentativa de reconhecimento."""
        ti = self._by_track.get(track_id)
        if ti is None:
            ti = TrackIdentity(track_id=track_id)
            self._by_track[track_id] = ti
        ti.last_attempt_frame = frame_index
        ti.attempts += 1

        if student_id is not None and score >= cfg.face_match_threshold:
            if ti.student_id == student_id:
                ti.hits += 1
            else:
                ti.student_id = student_id
                ti.hits = 1
            ti.score = score
            if ti.hits >= cfg.face_confirm_hits or score >= cfg.face_confirm_score:
                ti.confirmed = True

    def identity_of(self, track_id: int) -> str | None:
        ti = self._by_track.get(track_id)
        return ti.student_id if ti else None

    def is_confirmed(self, track_id: int) -> bool:
        ti = self._by_track.get(track_id)
        return bool(ti and ti.confirmed)
