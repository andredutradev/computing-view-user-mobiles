"""Painel de controle desktop (Tkinter) — opcional, import local.

Resolve dois pedidos de uma vez:
  - "subir um vídeo e abrir a janela como se fosse a câmera ao vivo": o botão
    "Procurar vídeo" escolhe um arquivo e "Iniciar" abre a janela OpenCV
    reproduzindo o vídeo anotado em tempo real (com tracking + suavização +
    reconhecimento de alunos);
  - "gerar e baixar o relatório a partir de uma interface": o botão
    "Gerar relatório" produz os arquivos PDF+CSV e abre a pasta de saída.

O ``Application.run()`` roda numa THREAD trabalhadora para o Tkinter não
congelar; a janela OpenCV e o ``waitKey`` vivem nessa thread. Um
``threading.Event`` permite o "Parar" encerrar o loop com segurança. Widgets do
Tk só são tocados via ``root.after`` (thread-safe).

Tudo aqui é stdlib (tkinter) + os módulos do projeto; lançado por
``python3 -m src.attendance.ui`` ou ``python3 -m src.main --ui``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.config import Config, settings


def _open_folder(path: Path) -> None:
    """Abre a pasta no gerenciador de arquivos do SO (best-effort)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":  # pragma: no cover - Windows
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:  # pragma: no cover - Linux
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


class ControlPanel:
    """Janela de controle do sistema de presença."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or settings
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._attendance = None
        self._session = None
        self._last_report_dir: Path | None = None
        self._status_text = "Pronto."
        self._counts = (0, 0, 0)  # (frame, pessoas, usando)

    # -- construção da UI ---------------------------------------------------
    def run(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self._tk = tk
        self._filedialog = filedialog
        self.root = tk.Tk()
        self.root.title("Computing View — Presença em Sala")
        self.root.geometry("560x520")

        pad = {"padx": 10, "pady": 4}
        row = 0

        ttk.Label(self.root, text="1) Matrícula de alunos", font=("", 12, "bold")).grid(
            column=0, row=row, columnspan=3, sticky="w", **pad
        )
        row += 1
        self.photos_var = tk.StringVar(value=self.config.students_dir)
        ttk.Entry(self.root, textvariable=self.photos_var, width=48).grid(
            column=0, row=row, columnspan=2, sticky="we", **pad
        )
        ttk.Button(self.root, text="Procurar...", command=self._pick_photos).grid(
            column=2, row=row, **pad
        )
        row += 1
        ttk.Button(
            self.root, text="Matricular alunos", command=self._enroll
        ).grid(column=0, row=row, sticky="w", **pad)
        row += 1

        ttk.Separator(self.root, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="we", pady=8
        )
        row += 1

        ttk.Label(self.root, text="2) Fonte de vídeo", font=("", 12, "bold")).grid(
            column=0, row=row, columnspan=3, sticky="w", **pad
        )
        row += 1
        self.source_var = tk.StringVar(value="file")
        ttk.Radiobutton(
            self.root, text="Arquivo de vídeo", value="file", variable=self.source_var
        ).grid(column=0, row=row, sticky="w", **pad)
        ttk.Radiobutton(
            self.root, text="Webcam", value="webcam", variable=self.source_var
        ).grid(column=1, row=row, sticky="w", **pad)
        row += 1
        self.video_var = tk.StringVar(value=self.config.video_path)
        ttk.Entry(self.root, textvariable=self.video_var, width=48).grid(
            column=0, row=row, columnspan=2, sticky="we", **pad
        )
        ttk.Button(self.root, text="Procurar vídeo...", command=self._pick_video).grid(
            column=2, row=row, **pad
        )
        row += 1
        ttk.Label(self.root, text="Índice da webcam:").grid(
            column=0, row=row, sticky="e", **pad
        )
        self.webcam_var = tk.IntVar(value=self.config.webcam_index)
        ttk.Spinbox(self.root, from_=0, to=10, textvariable=self.webcam_var, width=5).grid(
            column=1, row=row, sticky="w", **pad
        )
        row += 1

        ttk.Label(self.root, text="Limiar de reconhecimento:").grid(
            column=0, row=row, sticky="e", **pad
        )
        self.thresh_var = tk.DoubleVar(value=self.config.face_match_threshold)
        ttk.Scale(
            self.root, from_=0.2, to=0.7, variable=self.thresh_var, orient="horizontal"
        ).grid(column=1, row=row, columnspan=2, sticky="we", **pad)
        row += 1

        ttk.Separator(self.root, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="we", pady=8
        )
        row += 1

        ttk.Label(self.root, text="3) Execução", font=("", 12, "bold")).grid(
            column=0, row=row, columnspan=3, sticky="w", **pad
        )
        row += 1
        self.start_btn = ttk.Button(self.root, text="Iniciar", command=self._start)
        self.start_btn.grid(column=0, row=row, sticky="w", **pad)
        self.stop_btn = ttk.Button(
            self.root, text="Parar", command=self._stop, state="disabled"
        )
        self.stop_btn.grid(column=1, row=row, sticky="w", **pad)
        self.report_btn = ttk.Button(
            self.root, text="Gerar relatório", command=self._generate_report, state="disabled"
        )
        self.report_btn.grid(column=2, row=row, **pad)
        row += 1

        self.status_var = tk.StringVar(value=self._status_text)
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken").grid(
            column=0, row=row, columnspan=3, sticky="we", padx=10, pady=12
        )
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        self._poll_status()
        self.root.mainloop()

    # -- callbacks de UI ----------------------------------------------------
    def _pick_photos(self) -> None:
        d = self._filedialog.askdirectory(initialdir=self.photos_var.get() or ".")
        if d:
            self.photos_var.set(d)

    def _pick_video(self) -> None:
        f = self._filedialog.askopenfilename(
            title="Escolha um vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")],
        )
        if f:
            self.video_var.set(f)
            self.source_var.set("file")

    def _enroll(self) -> None:
        self._set_status("Matriculando alunos...")

        def task():
            from src.attendance.enrollment import enroll_directory
            from src.attendance.face_recognizer import FaceRecognizer

            cfg = self.config
            recognizer = FaceRecognizer(config=cfg)
            gallery, warnings = enroll_directory(
                self.photos_var.get(),
                recognizer,
                gallery_path=cfg.gallery_path,
                enrolled_at=datetime.now().isoformat(timespec="seconds"),
                config=cfg,
            )
            msg = f"{len(gallery)} aluno(s) matriculado(s)."
            if warnings:
                msg += f" {len(warnings)} aviso(s)."
            self._set_status(msg)

        threading.Thread(target=task, daemon=True).start()

    def _build_config(self) -> Config:
        """Config final com os overrides escolhidos na UI."""
        overrides = {
            "attendance_enabled": True,
            "face_match_threshold": float(self.thresh_var.get()),
            "video_source_type": self.source_var.get(),
            "video_path": self.video_var.get(),
            "webcam_index": int(self.webcam_var.get()),
        }
        return replace(self.config, **overrides)

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._session = None
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.report_btn.config(state="disabled")
        self._set_status("Iniciando... (a janela de vídeo vai abrir)")
        cfg = self._build_config()
        self._worker = threading.Thread(target=self._run_pipeline, args=(cfg,), daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self._set_status("Parando...")

    def _run_pipeline(self, cfg: Config) -> None:
        try:
            from src.attendance.attendance import AttendanceTracker
            from src.attendance.enrollment import Gallery
            from src.attendance.face_recognizer import FaceRecognizer
            from src.detector import Detector
            from src.main import Application
            from src.temporal import TemporalTracker
            from src.video_source import VideoSourceFactory
            from src.visualizer import Visualizer

            source = VideoSourceFactory.create(cfg)
            fps = getattr(source, "fps", float(cfg.output_fps))
            detector = Detector(config=cfg)
            visualizer = Visualizer(config=cfg)
            tracking = cfg.tracking_enabled and cfg.pose_enabled
            temporal = (
                TemporalTracker(cfg)
                if tracking and cfg.smoothing_enabled
                else None
            )
            gallery = Gallery.load(cfg.gallery_path)
            recognizer = FaceRecognizer(config=cfg)
            label = (
                Path(cfg.video_path).name
                if cfg.video_source_type == "file"
                else f"webcam_{cfg.webcam_index}"
            )
            tracker = AttendanceTracker(
                config=cfg,
                gallery=gallery,
                recognizer=recognizer,
                fps=fps,
                source_label=label,
                started_at=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            )
            self._attendance = tracker
            app = Application(
                cfg,
                source,
                detector,
                visualizer,
                show_window=True,
                tracking=tracking,
                temporal=temporal,
                attendance_tracker=tracker,
                stop_event=self._stop_event,
                on_frame=self._on_frame,
            )
            app.run()
        except Exception as exc:  # mostra o erro na barra de status
            self._set_status(f"Erro: {exc}")
        finally:
            if self._attendance is not None:
                self._session = self._attendance.finalize(
                    ended_at=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                )
            self.root.after(0, self._on_run_done)

    def _on_frame(self, annotated, people, frame_index) -> None:
        using = sum(1 for p in people if getattr(p, "using_phone", False))
        self._counts = (frame_index, len(people), using)

    def _on_run_done(self) -> None:
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if self._session is not None and self._session.students:
            self.report_btn.config(state="normal")
        self._set_status("Sessão encerrada. Você já pode gerar o relatório.")

    def _generate_report(self) -> None:
        if self._session is None:
            self._set_status("Nenhuma sessão para relatar.")
            return
        self._set_status("Gerando relatório...")

        def task():
            from src.attendance.reports import generate_reports

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths = generate_reports(
                self._session,
                out_dir=self.config.reports_dir,
                stamp=stamp,
                config=self.config,
            )
            folder = Path(paths[0]).parent if paths else Path(self.config.reports_dir)
            self._last_report_dir = folder
            self._set_status(f"Relatório salvo em {folder}")
            _open_folder(folder)

        threading.Thread(target=task, daemon=True).start()

    # -- utilidades thread-safe --------------------------------------------
    def _set_status(self, text: str) -> None:
        self._status_text = text

    def _poll_status(self) -> None:
        frame, people, using = self._counts
        suffix = (
            f"  |  frame {frame} · pessoas {people} · usando {using}"
            if frame
            else ""
        )
        self.status_var.set(self._status_text + suffix)
        self.root.after(200, self._poll_status)


def _tk_renders_ok() -> bool:
    """True se o Tk disponível renderiza de verdade (>= 8.6).

    No macOS recente o Tk 8.5 do sistema abre a janela toda PRETA. Detectamos
    isso para cair, automaticamente, no painel OpenCV (Cocoa nativo), que
    renderiza igual à janela do vídeo.
    """
    try:
        import tkinter

        return float(tkinter.TkVersion) >= 8.6
    except Exception:
        return False


def _pyside6_available() -> bool:
    """True se o PySide6 puder ser importado (backend Qt disponível)."""
    try:
        import importlib.util

        return importlib.util.find_spec("PySide6") is not None
    except Exception:  # pragma: no cover - defensivo
        return False


def run_ui(config: Config | None = None) -> int:
    """Abre o painel de controle. Devolve 0 ao fechar.

    Backend pela env ``CVUM_UI_BACKEND`` (``qt``/``tk``/``cv``/``auto``).
    Em ``auto`` (padrão) a ordem de preferência é:
      1. **Qt (PySide6)** — moderno, acentos nativos, vídeo embutido (recomendado);
      2. **Tkinter** — se o Tk renderizar bem (>= 8.6);
      3. **OpenCV** — último recurso (ex.: Tk 8.5 do macOS e sem PySide6).
    """
    cfg = config or settings
    backend = os.environ.get("CVUM_UI_BACKEND", "auto").lower()

    if backend == "qt" or (backend == "auto" and _pyside6_available()):
        try:
            from src.attendance.qt_panel import run_qt_panel

            return run_qt_panel(cfg)
        except ImportError as exc:
            if backend == "qt":
                print(f"[ERRO] PySide6 indisponível: {exc}")
                return 1
            print(f"[INFO] PySide6 indisponível ({exc}); tentando outro backend.")

    use_cv = backend == "cv" or (backend == "auto" and not _tk_renders_ok())
    if use_cv:
        if backend == "auto":
            print(
                "[INFO] Sem PySide6 e Tk do sistema antigo (janela preta no "
                "macOS); usando o painel OpenCV."
            )
        from src.attendance.cv_panel import run_cv_panel

        return run_cv_panel(cfg)

    ControlPanel(cfg).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_ui())
