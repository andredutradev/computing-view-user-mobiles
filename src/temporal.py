"""Suavização temporal do estado "usando celular" (consistência entre frames).

Problema que resolve
--------------------
A detecção do ``Detector`` é *stateless* (decide frame a frame). Como o YOLO e a
geometria oscilam um pouco a cada quadro — sobretudo com a pessoa se mexendo,
mãos entrando/saindo de oclusão, ou o celular piscando na detecção — o estado
"usando celular" alterna laranja↔verde rapidamente e pode "se perder" mesmo
quando a pessoa segue usando o aparelho.

Como resolve
------------
Esta camada é *stateful* e roda DEPOIS do Detector, mantendo um histórico por
``track_id`` (vindo do rastreamento ByteTrack/BoT-SORT). Para cada track:

  1. Voto em janela deslizante: empilha a decisão crua (``using_phone``) dos
     últimos N frames e calcula a fração de votos positivos (``ratio``).
  2. Histerese (dois limiares): LIGA o estado só quando ``ratio >= on_threshold``
     e DESLIGA só quando ``ratio <= off_threshold`` (com ``off < on``). Essa
     "zona morta" entre os limiares é o que elimina o tremido (flicker).
  3. Grace period: um track que some por poucos frames (oclusão/movimento)
     continua sendo emitido com seu último estado, em vez de desaparecer. Após
     ``track_max_age`` frames sem ser visto, o histórico é descartado (memória
     limitada).

Pessoas sem ``track_id`` (tracking desligado, cold-start do tracker, ou caminho
``--no-pose``) passam direto, sem suavização — nunca quebram.

Mantém o ``Detector`` puro/stateless: todo o estado mutável vive AQUI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.config import Config, settings
from src.detector import PersonDetection


@dataclass
class TrackState:
    """Histórico de um track (uma pessoa) ao longo dos frames."""

    track_id: int
    # Votos crus (using_phone) dos últimos frames; maxlen = smoothing_window.
    votes: deque = field(default_factory=deque)
    # Estado suavizado atual (saída da histerese).
    smoothed_using: bool = False
    # Último frame em que o track foi efetivamente visto.
    last_seen_frame: int = 0
    # Última PersonDetection boa — reusada para "fantasmas" no grace period.
    last_person: PersonDetection | None = None


class TemporalTracker:
    """Aplica voto deslizante + histerese + grace period por ``track_id``."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or settings
        self._tracks: dict[int, TrackState] = {}

    # -- API pública --------------------------------------------------------
    def update(
        self, people: list[PersonDetection], frame_index: int
    ) -> list[PersonDetection]:
        """Recebe as pessoas CRUAS do frame e devolve com o estado SUAVIZADO.

        Também reintroduz tracks "em grace" (vistos há poucos frames) para que
        o box não pisque/suma durante oclusões breves.
        """
        cfg = self.config
        out: list[PersonDetection] = []
        seen_ids: set[int] = set()

        for person in people:
            tid = person.track_id
            if tid is None:
                # Sem ID estável: não dá para acumular histórico -> passa cru.
                out.append(person)
                continue

            seen_ids.add(tid)
            state = self._tracks.get(tid)
            if state is None:
                state = TrackState(
                    track_id=tid,
                    votes=deque(maxlen=max(1, cfg.smoothing_window)),
                )
                self._tracks[tid] = state

            state.votes.append(bool(person.using_phone))
            state.last_seen_frame = frame_index
            state.smoothed_using = self._apply_hysteresis(state)

            # Sobrescreve a decisão crua pela suavizada antes de devolver.
            person.using_phone = state.smoothed_using
            state.last_person = person
            out.append(person)

        # Grace / evicção: trata tracks não vistos neste frame.
        for tid, state in list(self._tracks.items()):
            if tid in seen_ids:
                continue
            age = frame_index - state.last_seen_frame
            if age > cfg.track_max_age:
                del self._tracks[tid]  # esquece de vez (limita memória)
                continue
            if age <= cfg.grace_period and state.last_person is not None:
                # Reemite a última detecção mantendo o estado suavizado, para
                # que o box "sobreviva" à ausência momentânea.
                ghost = state.last_person
                ghost.using_phone = state.smoothed_using
                out.append(ghost)

        return out

    # -- internos -----------------------------------------------------------
    def _apply_hysteresis(self, state: TrackState) -> bool:
        """Decide o estado suavizado a partir da fração de votos positivos.

        Exige um mínimo de amostras antes de LIGAR pela primeira vez, para que
        um único frame ruidoso no início não dispare o estado.
        """
        cfg = self.config
        n = len(state.votes)
        if n == 0:
            return state.smoothed_using

        ratio = sum(state.votes) / n
        # Espera encher um pouco a janela antes de confiar para LIGAR.
        warmup = min(3, cfg.smoothing_window)

        if state.smoothed_using:
            # Já LIGADO: só desliga ao cair até/abaixo do limiar inferior.
            if ratio <= cfg.off_threshold:
                return False
            return True
        # DESLIGADO: liga ao atingir o limiar superior (após o warmup).
        if n >= warmup and ratio >= cfg.on_threshold:
            return True
        return False

    # -- introspecção (útil para testes/depuração) -------------------------
    @property
    def active_track_count(self) -> int:
        """Quantidade de tracks atualmente em memória."""
        return len(self._tracks)
