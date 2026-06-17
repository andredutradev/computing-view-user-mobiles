"""Modelo de dados da sessão de monitoramento (presença + ocupação).

Conceitos:
  - ``StudentSession``: acúmulo de PRESENÇA de UM aluno em intervalos
    ``(entrada, saída)``. Sumiços curtos (<= grace) NÃO fecham o intervalo
    (são "ponte"); sumiços longos fecham o intervalo e uma reaparição abre um
    novo (re-entrada). O tempo total de permanência (dwell) é derivado da soma
    dos intervalos.
  - ``OccupancySample``: contagem de pessoas/identificados/usando-celular em um
    instante — a série temporal de OCUPAÇÃO.
  - ``Session``: agrega os alunos + a série de ocupação + metadados.

O tempo é em SEGUNDOS de vídeo (``frame_index / fps``), o que torna os
relatórios reproduzíveis a partir de um arquivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OccupancySample:
    """Uma amostra instantânea de ocupação da sala."""

    frame_index: int
    t_seconds: float
    people_count: int
    identified_count: int
    using_phone_count: int


@dataclass
class StudentSession:
    """Presença acumulada de um aluno, em intervalos (entrada, saída)."""

    student_id: str
    display_name: str
    intervals: list = field(default_factory=list)  # list[tuple[float, float]]
    _open_start: float | None = field(default=None)
    _last_t: float | None = field(default=None)
    # Uso de celular: mesmos intervalos (entrada, saída), mas só nos instantes
    # em que a pessoa estava SEGURANDO o aparelho. Independe da presença.
    phone_intervals: list = field(default_factory=list)
    _phone_open: float | None = field(default=None)
    _phone_last: float | None = field(default=None)

    def mark_present(self, t: float, grace_seconds: float) -> None:
        """Registra que o aluno está presente no instante ``t`` (segundos)."""
        if self._open_start is None:
            self._open_start = t
            self._last_t = t
            return
        gap = t - (self._last_t if self._last_t is not None else t)
        if gap > grace_seconds:
            # Sumiço longo: fecha o intervalo anterior e abre um novo.
            self.intervals.append((self._open_start, self._last_t))
            self._open_start = t
        # gap <= grace: ponte (mantém o intervalo aberto, conta o vão).
        self._last_t = t

    def mark_phone_use(self, t: float, grace_seconds: float) -> None:
        """Registra USO de celular no instante ``t`` (mesma lógica de ponte)."""
        if self._phone_open is None:
            self._phone_open = t
            self._phone_last = t
            return
        gap = t - (self._phone_last if self._phone_last is not None else t)
        if gap > grace_seconds:
            self.phone_intervals.append((self._phone_open, self._phone_last))
            self._phone_open = t
        self._phone_last = t

    def finalize(self) -> None:
        """Fecha os intervalos abertos (presença e celular)."""
        if self._open_start is not None and self._last_t is not None:
            self.intervals.append((self._open_start, self._last_t))
            self._open_start = None
            self._last_t = None
        if self._phone_open is not None and self._phone_last is not None:
            self.phone_intervals.append((self._phone_open, self._phone_last))
            self._phone_open = None
            self._phone_last = None

    @property
    def present_seconds(self) -> float:
        """Tempo total de permanência (soma dos intervalos + o aberto)."""
        total = sum(end - start for start, end in self.intervals)
        if self._open_start is not None and self._last_t is not None:
            total += self._last_t - self._open_start
        return float(total)

    @property
    def phone_seconds(self) -> float:
        """Tempo total SEGURANDO o celular (soma dos intervalos + o aberto)."""
        total = sum(end - start for start, end in self.phone_intervals)
        if self._phone_open is not None and self._phone_last is not None:
            total += self._phone_last - self._phone_open
        return float(total)

    @property
    def first_seen_t(self) -> float | None:
        if self.intervals:
            return self.intervals[0][0]
        return self._open_start

    @property
    def last_seen_t(self) -> float | None:
        if self._open_start is not None:
            return self._last_t
        if self.intervals:
            return self.intervals[-1][1]
        return None

    @property
    def num_intervals(self) -> int:
        n = len(self.intervals)
        return n + (1 if self._open_start is not None else 0)


@dataclass
class Session:
    """Agregado da sessão: alunos, ocupação ao longo do tempo e metadados."""

    fps: float
    source_label: str = "webcam"
    started_t: float = 0.0
    # Data/hora reais (wall-clock) de início e fim do monitoramento, já
    # formatadas pelo chamador (ex.: "17/06/2026 14:30:00"). Vazio se não
    # informado — a stdlib de datas é injetada de fora para manter testabilidade.
    started_at: str = ""
    ended_at: str = ""
    students: dict = field(default_factory=dict)  # dict[str, StudentSession]
    occupancy: list = field(default_factory=list)  # list[OccupancySample]
    last_t: float = 0.0

    def ensure_student(self, student_id: str, display_name: str) -> StudentSession:
        st = self.students.get(student_id)
        if st is None:
            st = StudentSession(student_id=student_id, display_name=display_name)
            self.students[student_id] = st
        return st

    def mark_present(
        self, student_id: str, display_name: str, t: float, grace_seconds: float
    ) -> None:
        self.ensure_student(student_id, display_name).mark_present(t, grace_seconds)
        self.last_t = max(self.last_t, t)

    def mark_phone_use(
        self, student_id: str, display_name: str, t: float, grace_seconds: float
    ) -> None:
        """Credita tempo de USO de celular para a pessoa (aluno ou anônimo)."""
        self.ensure_student(student_id, display_name).mark_phone_use(t, grace_seconds)
        self.last_t = max(self.last_t, t)

    def add_occupancy(self, sample: OccupancySample) -> None:
        self.occupancy.append(sample)
        self.last_t = max(self.last_t, sample.t_seconds)

    def finalize(self) -> None:
        for st in self.students.values():
            st.finalize()

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_t - self.started_t)

    def peak_occupancy(self) -> tuple:
        """(t_seconds, contagem) do pico de ocupação. (0,0) se vazio."""
        if not self.occupancy:
            return (0.0, 0)
        peak = max(self.occupancy, key=lambda s: s.people_count)
        return (peak.t_seconds, peak.people_count)

    def average_occupancy(self) -> float:
        if not self.occupancy:
            return 0.0
        return sum(s.people_count for s in self.occupancy) / len(self.occupancy)
