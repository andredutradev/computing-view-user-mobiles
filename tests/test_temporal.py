"""Testes da suavização temporal (``src/temporal.py``).

Estratégia: NÃO depende de YOLO nem de torch. Construímos ``PersonDetection``
diretamente (como em ``test_detector.py``) e alimentamos sequências booleanas
determinísticas no ``TemporalTracker``, verificando o frame EXATO de cada
transição de estado, o grace period e a evicção por idade.
"""

from __future__ import annotations

from src.config import Config
from src.detector import PersonDetection
from src.temporal import TemporalTracker


def _person(track_id, using: bool) -> PersonDetection:
    """Cria uma pessoa mínima com track_id e estado cru de uso de celular."""
    return PersonDetection(
        class_id=0,
        confidence=0.9,
        box=(0.0, 0.0, 10.0, 100.0),
        using_phone=using,
        track_id=track_id,
    )


def _smoothed_state_for(out, track_id):
    """Extrai o using_phone suavizado do track pedido na saída do update()."""
    for p in out:
        if p.track_id == track_id:
            return p.using_phone
    return None  # track ausente neste frame (nem visto, nem fantasma)


# Config pequena e determinística para os testes de histerese.
def _cfg() -> Config:
    return Config(
        smoothing_window=5,
        on_threshold=0.6,
        off_threshold=0.35,
        grace_period=2,
        track_max_age=5,
    )


def test_hysteresis_on_off_transition_frames():
    """T,T,T (liga após warmup) e depois F... (desliga só sob off_threshold)."""
    tracker = TemporalTracker(_cfg())
    # Sequência de decisões CRUAS por frame para o track 1.
    raw = [True, True, True, False, False, False, False]
    # Estado SUAVIZADO esperado por frame (ver cálculo no plano):
    #   warmup=3; liga no frame 2 (ratio 1.0>=0.6); mantém ligado enquanto
    #   ratio>0.35; desliga no frame 6 (ratio 1/5=0.2<=0.35).
    expected = [False, False, True, True, True, True, False]

    got = []
    for i, r in enumerate(raw):
        out = tracker.update([_person(1, r)], frame_index=i)
        got.append(_smoothed_state_for(out, 1))

    assert got == expected


def test_does_not_flip_on_before_warmup():
    """Um único frame positivo no início não deve ligar o estado."""
    tracker = TemporalTracker(_cfg())
    out = tracker.update([_person(1, True)], frame_index=0)
    assert _smoothed_state_for(out, 1) is False


def test_grace_period_keeps_state_during_brief_absence():
    """Track sumindo por <= grace continua emitido (fantasma) com estado mantido."""
    tracker = TemporalTracker(_cfg())
    # Liga o estado (frames 0,1,2 -> ON no frame 2).
    for i in range(3):
        tracker.update([_person(1, True)], frame_index=i)

    # Frame 3: track ausente, age=1 <= grace(2) -> fantasma emitido, ON.
    out3 = tracker.update([], frame_index=3)
    assert _smoothed_state_for(out3, 1) is True

    # Frame 4: age=2 <= grace(2) -> ainda fantasma.
    out4 = tracker.update([], frame_index=4)
    assert _smoothed_state_for(out4, 1) is True

    # Frame 5: age=3 > grace(2) -> não emite mais o fantasma.
    out5 = tracker.update([], frame_index=5)
    assert _smoothed_state_for(out5, 1) is None


def test_track_evicted_after_max_age():
    """Após track_max_age frames sem ser visto, o histórico é descartado."""
    tracker = TemporalTracker(_cfg())
    tracker.update([_person(1, True)], frame_index=0)  # cria o track
    assert tracker.active_track_count == 1

    # last_seen=0, track_max_age=5 -> deletado quando age>5 (frame_index>=6).
    for i in range(1, 6):
        tracker.update([], frame_index=i)
        assert tracker.active_track_count == 1  # ainda em memória (age<=5)

    tracker.update([], frame_index=6)  # age=6 > 5 -> esquece
    assert tracker.active_track_count == 0


def test_untracked_person_passes_through_raw():
    """Pessoa sem track_id passa crua (sem suavização) e não cria estado."""
    tracker = TemporalTracker(_cfg())
    out = tracker.update([_person(None, True)], frame_index=0)
    assert len(out) == 1
    assert out[0].using_phone is True  # crua, inalterada
    assert tracker.active_track_count == 0


def test_independent_tracks_do_not_interfere():
    """Dois tracks mantêm históricos separados."""
    tracker = TemporalTracker(_cfg())
    # Track 1 sempre usando; track 2 nunca.
    for i in range(5):
        out = tracker.update(
            [_person(1, True), _person(2, False)], frame_index=i
        )
    assert _smoothed_state_for(out, 1) is True
    assert _smoothed_state_for(out, 2) is False
    assert tracker.active_track_count == 2
