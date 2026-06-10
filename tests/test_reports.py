"""Testes de geração de relatórios (CSV sempre; PDF se fpdf2 estiver instalado)."""

from __future__ import annotations

import csv

import pytest

from src.attendance.reports import attendance_rows, generate_reports, hms
from src.attendance.session import OccupancySample, Session
from src.config import Config


def _session_with_data():
    s = Session(fps=10.0, source_label="aula1")
    # Ana: presente ~2s (ponte de gap curto) e usando o celular ~1s.
    s.ensure_student("ana", "Ana")
    s.mark_present("ana", "Ana", 0.0, 3.0)
    s.mark_present("ana", "Ana", 2.0, 3.0)
    s.mark_phone_use("ana", "Ana", 0.0, 3.0)
    s.mark_phone_use("ana", "Ana", 1.0, 3.0)
    # Bruno: matriculado, nunca visto -> ausente.
    s.ensure_student("bruno", "Bruno")
    s.add_occupancy(OccupancySample(0, 0.0, 1, 1, 0))
    s.add_occupancy(OccupancySample(20, 2.0, 1, 1, 1))
    s.finalize()
    return s


def test_hms_format():
    assert hms(0) == "00:00:00"
    assert hms(3661) == "01:01:01"


def test_attendance_rows_present_and_absent():
    cfg = Config(attendance_min_seconds=1.0)
    rows = {r["student_id"]: r for r in attendance_rows(_session_with_data(), cfg)}
    assert rows["ana"]["present"] == "sim"
    assert rows["ana"]["total_seconds"] >= 2.0
    assert rows["ana"]["phone_seconds"] == pytest.approx(1.0)
    assert rows["bruno"]["present"] == "nao"
    assert rows["bruno"]["total_seconds"] == 0.0
    assert rows["bruno"]["phone_seconds"] == 0.0


def test_generate_reports_writes_files(tmp_path):
    cfg = Config(attendance_min_seconds=1.0)
    paths = generate_reports(
        _session_with_data(), out_dir=tmp_path, stamp="t", config=cfg
    )
    names = {p.name for p in paths}
    assert "frequencia.csv" in names
    assert "ocupacao.csv" in names

    freq = next(p for p in paths if p.name == "frequencia.csv")
    with open(freq, newline="", encoding="utf-8") as fh:
        data = {row["student_id"]: row for row in csv.DictReader(fh)}
    assert data["ana"]["present"] == "sim"
    assert data["ana"]["phone_hms"] == "00:00:01"
    assert "bruno" in data
