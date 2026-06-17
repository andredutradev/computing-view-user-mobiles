"""Presença e movimentação em tempo real, por ``track_id``.

``AttendanceTracker`` é o orquestrador do subsistema de presença. A cada frame
recebe o frame e a lista de ``PersonDetection`` já rastreadas (com ``track_id``
estável vindo da Fase de tracking) e:

  1. mantém um ``TrackState`` por track (movimentação, primeiro/último frame);
  2. dispara o reconhecimento facial de forma amortizada (via ``IdentityCache``):
     só em tracks não confirmados e a cada N frames;
  3. credita PRESENÇA do aluno identificado no ``Session`` (dwell time);
  4. registra uma amostra de OCUPAÇÃO por frame.

Consome o ``track_id`` produzido pelo tracking (ByteTrack). Quando uma pessoa
vem sem ``track_id`` (cold-start), ela conta para a ocupação, mas não para o
dwell por aluno.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.attendance.face_recognizer import IdentityCache
from src.attendance.geometry import crop_box, head_roi_from_keypoints
from src.attendance.session import OccupancySample, Session
from src.config import Config, settings


def _now_str() -> str:
    """Data/hora atual formatada (dd/mm/aaaa HH:MM:SS) p/ o relatório."""
    from datetime import datetime

    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


class MovementState(str, Enum):
    SEATED = "sentado"
    MOVING = "movimentando"
    PRESENT = "presente"
    ABSENT = "ausente"


@dataclass
class TrackState:
    """Estado por track para presença/movimentação."""

    track_id: int
    student_id: str | None = None
    first_frame: int = 0
    last_frame: int = 0
    last_center: tuple | None = field(default=None)
    smoothed_speed: float = 0.0
    seated_frames: int = 0
    movement: MovementState = MovementState.PRESENT


class AttendanceTracker:
    """Acompanha presença, identidade e movimentação ao longo da sessão."""

    def __init__(
        self,
        config: Config | None = None,
        gallery=None,
        recognizer=None,
        fps: float = 25.0,
        source_label: str = "webcam",
        started_at: str = "",
    ) -> None:
        self.config = config or settings
        self.gallery = gallery
        self.recognizer = recognizer
        self.fps = float(fps) if fps and fps > 0 else float(self.config.output_fps)
        # Início real (wall-clock) do monitoramento: usa o que o chamador injetar
        # ou, na falta, captura agora — assim o relatório sempre registra a data.
        self.session = Session(
            fps=self.fps,
            source_label=source_label,
            started_at=started_at or _now_str(),
        )
        self.identity = IdentityCache()
        self._tracks: dict = {}  # dict[int, TrackState]

        # Pré-registra os alunos matriculados, para que ausentes apareçam no
        # relatório (present=False) — uma chamada importante para a frequência.
        if gallery is not None:
            for sid, rec in gallery.records.items():
                self.session.ensure_student(sid, rec.display_name)

    # -- loop por frame -----------------------------------------------------
    def update(self, frame: np.ndarray, people: list, frame_index: int) -> None:
        cfg = self.config
        t = frame_index / self.fps

        people_count = 0
        identified_count = 0
        using_count = 0

        for person in people:
            people_count += 1
            if person.using_phone:
                using_count += 1

            tid = person.track_id
            if tid is None:
                continue  # sem ID estável: conta ocupação, não dwell

            state = self._tracks.get(tid)
            if state is None:
                state = TrackState(track_id=tid, first_frame=frame_index)
                self._tracks[tid] = state
            state.last_frame = frame_index
            self._update_movement(state, person)

            # Reconhecimento facial amortizado.
            if self.recognizer is not None and self.gallery is not None:
                if self.identity.should_recognize(tid, frame_index, cfg):
                    sid, score = self._recognize(frame, person)
                    self.identity.update(tid, sid, score, frame_index, cfg)
                state.student_id = self.identity.identity_of(tid)

            # Crédito de presença para o aluno identificado.
            sid = state.student_id
            if sid is not None:
                identified_count += 1
                name = (
                    self.gallery.records[sid].display_name
                    if (self.gallery and sid in self.gallery.records)
                    else sid
                )
                self.session.mark_present(sid, name, t, cfg.presence_grace_seconds)

            # Crédito de TEMPO DE CELULAR por pessoa. Vale para o aluno
            # identificado e também para tracks anônimos (Pessoa #ID), para que
            # ninguém que use o celular fique de fora do relatório.
            if person.using_phone:
                if sid is not None:
                    phone_key, phone_name = sid, name
                else:
                    phone_key, phone_name = f"anon_{tid}", f"Pessoa #{tid}"
                self.session.mark_phone_use(
                    phone_key, phone_name, t, cfg.presence_grace_seconds
                )

        self.session.add_occupancy(
            OccupancySample(
                frame_index=frame_index,
                t_seconds=t,
                people_count=people_count,
                identified_count=identified_count,
                using_phone_count=using_count,
            )
        )

    # -- visual -------------------------------------------------------------
    def identity_labels(self) -> dict:
        """Mapa ``track_id -> nome`` para o Visualizer (só identificados)."""
        out: dict = {}
        for tid, state in self._tracks.items():
            if state.student_id is None:
                continue
            if self.gallery and state.student_id in self.gallery.records:
                out[tid] = self.gallery.records[state.student_id].display_name
            else:
                out[tid] = state.student_id
        return out

    # -- internos -----------------------------------------------------------
    def _recognize(self, frame: np.ndarray, person):
        """Recorta a cabeça da pessoa e reconhece contra a galeria."""
        if frame is None:
            return (None, 0.0)
        roi = head_roi_from_keypoints(person.keypoints, person.box, self.config)
        crop = crop_box(frame, roi)
        if crop is None:
            return (None, 0.0)
        h, w = crop.shape[:2]
        if min(h, w) < self.config.face_min_size_px:
            # Rosto muito pequeno no recorte: tenta a caixa inteira da pessoa
            # (fallback). Comparação explícita com None — arrays numpy não
            # podem ser avaliados como booleanos.
            bigger = crop_box(frame, person.box)
            if bigger is not None:
                crop = bigger
        emb = self.recognizer.embed_face(crop)
        if emb is None:
            return (None, 0.0)
        return self.recognizer.match(emb, self.gallery, self.config.face_match_threshold)

    def _update_movement(self, state: TrackState, person) -> None:
        cfg = self.config
        x1, y1, x2, y2 = person.box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        height = max(1.0, y2 - y1)
        if state.last_center is not None:
            disp = float(
                np.hypot(cx - state.last_center[0], cy - state.last_center[1])
            ) / height
            # EMA para suavizar a velocidade (ruído de detecção).
            state.smoothed_speed = 0.6 * state.smoothed_speed + 0.4 * disp
        state.last_center = (cx, cy)

        if state.smoothed_speed >= cfg.move_speed_threshold:
            state.movement = MovementState.MOVING
            state.seated_frames = 0
        elif state.smoothed_speed <= cfg.seated_speed_threshold:
            state.seated_frames += 1
            state.movement = (
                MovementState.SEATED
                if state.seated_frames >= cfg.seated_min_frames
                else MovementState.PRESENT
            )
        else:
            state.movement = MovementState.PRESENT
            state.seated_frames = 0

    # -- finalização --------------------------------------------------------
    def finalize(self, ended_at: str = "") -> Session:
        """Fecha os intervalos abertos e devolve a sessão pronta p/ relatório.

        ``ended_at`` (data/hora real do fim, já formatada) é gravado na sessão
        para constar no relatório; vazio captura o instante atual.
        """
        self.session.ended_at = ended_at or _now_str()
        self.session.finalize()
        return self.session
