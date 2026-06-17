"""Geração de relatórios de frequência e ocupação (CSV + PDF, baixáveis).

Dois relatórios, cada um em CSV (stdlib ``csv``) e PDF (``fpdf2`` — import
local, pure-Python, leve):

  - FREQUÊNCIA (por aluno): presente?, tempo total, primeiro/último avistamento,
    % da sessão e nº de intervalos. Alunos matriculados nunca vistos aparecem
    com ``presente=False`` (ausências explícitas no diário de classe).
  - OCUPAÇÃO (série temporal + resumo): contagem de pessoas/identificados/
    usando-celular por instante, mais pico e média de ocupação.

``generate_reports`` grava os 4 arquivos em ``<reports_dir>/<fonte>_<stamp>/`` e
devolve os caminhos.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from src.attendance.session import Session
from src.config import Config, settings


def hms(seconds: float) -> str:
    """Formata segundos como HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_") or "fonte"


def attendance_rows(session: Session, cfg: Config) -> list:
    """Linhas do relatório de frequência (uma por aluno matriculado/visto)."""
    duration = session.duration_seconds or 0.0
    rows = []
    for sid, st in sorted(session.students.items()):
        present_s = st.present_seconds
        # Aluno CADASTRADO (id não-anônimo) identificado em qualquer instante do
        # monitoramento conta como PRESENTE: a identidade facial é sinal forte e
        # dispensa o limiar de tempo mínimo (que serve só p/ tracks anônimos).
        is_enrolled = not sid.startswith("anon_")
        seen = st.first_seen_t is not None
        present = (is_enrolled and seen) or (present_s >= cfg.attendance_min_seconds)
        pct = (present_s / duration * 100.0) if duration > 0 else 0.0
        phone_s = st.phone_seconds
        # % do tempo de celular sobre a permanência (quanto da presença foi no
        # celular); cai para a duração da sessão se a pessoa não tem presença
        # creditada (ex.: track anônimo só com tempo de celular).
        phone_base = present_s if present_s > 0 else duration
        pct_phone = (phone_s / phone_base * 100.0) if phone_base > 0 else 0.0
        rows.append(
            {
                "student_id": sid,
                "display_name": st.display_name,
                "present": "sim" if present else "nao",
                # Ícone para leitura humana (CSV/planilha): check se presente, x se ausente.
                "presenca": "✓" if present else "x",
                "total_seconds": round(present_s, 1),
                "total_hms": hms(present_s),
                "phone_seconds": round(phone_s, 1),
                "phone_hms": hms(phone_s),
                "pct_phone": round(pct_phone, 1),
                "first_seen": hms(st.first_seen_t) if st.first_seen_t is not None else "-",
                "last_seen": hms(st.last_seen_t) if st.last_seen_t is not None else "-",
                "pct_session": round(pct, 1),
                "num_intervals": st.num_intervals,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_attendance_csv(session: Session, path: Path, cfg: Config) -> Path:
    rows = attendance_rows(session, cfg)
    fields = [
        "student_id",
        "display_name",
        "presenca",
        "present",
        "total_seconds",
        "total_hms",
        "phone_seconds",
        "phone_hms",
        "pct_phone",
        "first_seen",
        "last_seen",
        "pct_session",
        "num_intervals",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_occupancy_csv(session: Session, path: Path) -> Path:
    fields = [
        "frame_index",
        "t_seconds",
        "timestamp_hms",
        "people_count",
        "identified_count",
        "using_phone_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in session.occupancy:
            writer.writerow(
                {
                    "frame_index": s.frame_index,
                    "t_seconds": round(s.t_seconds, 3),
                    "timestamp_hms": hms(s.t_seconds),
                    "people_count": s.people_count,
                    "identified_count": s.identified_count,
                    "using_phone_count": s.using_phone_count,
                }
            )
    return path


# ---------------------------------------------------------------------------
# PDF (fpdf2)
# ---------------------------------------------------------------------------
def _new_pdf():
    """Cria um documento FPDF (import local)."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    return pdf


def _safe(text: str) -> str:
    """Latin-1 seguro para as fontes core do fpdf2 (substitui o que faltar)."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _ln(pdf, h: float, text: str) -> None:
    """Escreve uma linha cheia e quebra (substitui o ``ln=1`` depreciado)."""
    from fpdf.enums import XPos, YPos

    pdf.cell(0, h, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _write_period(pdf, session: Session) -> None:
    """Escreve início/fim do monitoramento e a duração (HH:MM:SS) no cabeçalho."""
    if session.started_at:
        _ln(pdf, 6, f"Início do monitoramento: {session.started_at}")
    if session.ended_at:
        _ln(pdf, 6, f"Fim do monitoramento: {session.ended_at}")
    _ln(pdf, 6, f"Duração: {hms(session.duration_seconds)}")


def _draw_presence(pdf, w: float, h: float, present: bool) -> None:
    """Desenha a célula de presença com ícone vetorial (check verde / x vermelho).

    As fontes core do fpdf2 são latin-1 e não têm o caractere de check (✓), então
    o ícone é traçado com linhas — sempre renderiza, independente da fonte.
    """
    x, y = pdf.get_x(), pdf.get_y()
    pdf.cell(w, h, "", border=1)  # célula vazia com borda (avança o cursor)
    cx, cy = x + w / 2, y + h / 2
    pdf.set_line_width(0.6)
    if present:
        pdf.set_draw_color(0, 150, 0)  # check verde
        pdf.line(cx - 2.2, cy + 0.2, cx - 0.6, cy + 1.8)
        pdf.line(cx - 0.6, cy + 1.8, cx + 2.4, cy - 1.8)
    else:
        pdf.set_draw_color(200, 0, 0)  # x vermelho
        pdf.line(cx - 1.8, cy - 1.8, cx + 1.8, cy + 1.8)
        pdf.line(cx - 1.8, cy + 1.8, cx + 1.8, cy - 1.8)
    pdf.set_draw_color(0, 0, 0)  # restaura padrões
    pdf.set_line_width(0.2)


def write_attendance_pdf(session: Session, path: Path, cfg: Config) -> Path:
    pdf = _new_pdf()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    _ln(pdf, 10, "Relatório de Frequência")
    pdf.set_font("Helvetica", "", 10)
    _ln(pdf, 6, f"Fonte: {session.source_label}")
    _write_period(pdf, session)
    present_count = sum(
        1 for r in attendance_rows(session, cfg) if r["present"] == "sim"
    )
    _ln(pdf, 6, f"Presentes: {present_count} / {len(session.students)}")
    pdf.ln(2)

    headers = [
        "Nome",
        "Presença",
        "Tempo à Vista",
        "Tempo Celular",
        "% Cel.",
        "Entrada",
        "Saída",
    ]
    widths = [46, 20, 27, 27, 16, 22, 22]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, _safe(h), border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in attendance_rows(session, cfg):
        pdf.cell(widths[0], 6, _safe(r["display_name"]), border=1)
        _draw_presence(pdf, widths[1], 6, r["present"] == "sim")
        pdf.cell(widths[2], 6, _safe(r["total_hms"]), border=1, align="C")
        pdf.cell(widths[3], 6, _safe(r["phone_hms"]), border=1, align="C")
        pdf.cell(widths[4], 6, _safe(f"{r['pct_phone']}%"), border=1, align="C")
        pdf.cell(widths[5], 6, _safe(r["first_seen"]), border=1, align="C")
        pdf.cell(widths[6], 6, _safe(r["last_seen"]), border=1, align="C")
        pdf.ln()
    pdf.output(str(path))
    return path


def write_occupancy_pdf(session: Session, path: Path) -> Path:
    pdf = _new_pdf()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    _ln(pdf, 10, "Relatório de Ocupação")
    pdf.set_font("Helvetica", "", 10)
    peak_t, peak_n = session.peak_occupancy()
    _ln(pdf, 6, f"Fonte: {session.source_label}")
    _write_period(pdf, session)
    _ln(pdf, 6, f"Ocupação de pico: {peak_n} (em {hms(peak_t)})")
    _ln(pdf, 6, f"Ocupação média: {session.average_occupancy():.1f}")
    _ln(pdf, 6, f"Amostras de ocupação: {len(session.occupancy)}")
    pdf.output(str(path))
    return path


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def generate_reports(
    session: Session,
    out_dir=None,
    stamp: str = "",
    config: Config | None = None,
) -> list:
    """Gera os 4 arquivos (CSV+PDF de frequência e ocupação).

    ``stamp`` (ex.: "20260607_143000") evita sobrescrever execuções; passe um
    valor pronto (a stdlib de datas é injetada pelo chamador). Devolve os
    caminhos gerados.
    """
    cfg = config or settings
    base = Path(out_dir or cfg.reports_dir)
    folder = base / f"{_slug(session.source_label)}_{stamp or 'sessao'}"
    folder.mkdir(parents=True, exist_ok=True)

    paths = [
        write_attendance_csv(session, folder / "frequencia.csv", cfg),
        write_occupancy_csv(session, folder / "ocupacao.csv"),
    ]
    # PDFs por último: se fpdf2 não estiver instalado, ainda entregamos os CSVs.
    try:
        paths.append(write_attendance_pdf(session, folder / "frequencia.pdf", cfg))
        paths.append(write_occupancy_pdf(session, folder / "ocupacao.pdf"))
    except ImportError:
        pass
    return paths
