"""Painel de controle baseado em OpenCV (sem Tkinter).

Alternativa ao ``ControlPanel`` (Tkinter) para ambientes onde o Tk do sistema é
antigo — no macOS recente, o **Tk 8.5** abre a janela toda **preta**. Este painel
usa a MESMA janela nativa (Cocoa) que o vídeo já usa, então renderiza de forma
confiável, e oferece os botões pedidos: **Matricular**, **Iniciar** (abre a
câmera anotada) e **Gerar relatório**.

Fluxo single-thread (sem worker/threads do Tk): clicar em "Iniciar" fecha o
painel, roda o pipeline ao vivo (a janela do vídeo bloqueia até 'q'/ESC) e, ao
terminar, o painel reaparece com "Gerar relatório" habilitado.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, settings
from src.text_render import TextItem, default_renderer

# Paleta (BGR) do painel.
_BG = (32, 32, 32)
_PANEL = (45, 45, 45)
_ACCENT = (0, 180, 0)        # verde: ação principal
_ACCENT_BLUE = (200, 140, 0)  # azul: ação secundária
_DISABLED = (70, 70, 70)
_TEXT = (240, 240, 240)
_MUTED = (160, 160, 160)

_W, _H = 720, 480


@dataclass
class _Button:
    key: str
    label: str
    box: tuple  # (x1, y1, x2, y2)
    color: tuple
    enabled: bool = True

    def hit(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.box
        return self.enabled and x1 <= x <= x2 and y1 <= y <= y2


class CvControlPanel:
    """Janela de controle desenhada com OpenCV (botões clicáveis)."""

    WIN = "Computing View — Painel de Controle"

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or settings
        self._session = None
        self._last_report_dir: Path | None = None
        self._status = "Pronto. Clique em INICIAR para abrir a câmera."
        self._click: tuple | None = None

    # -- mouse --------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    # -- desenho ------------------------------------------------------------
    def _buttons(self) -> list:
        has_session = self._session is not None and bool(
            getattr(self._session, "students", None)
        )
        return [
            _Button("enroll", "MATRICULAR alunos (data/students)", (40, 150, 680, 200), _ACCENT_BLUE),
            _Button("start", "INICIAR  (abre a câmera)", (40, 220, 680, 280), _ACCENT),
            _Button(
                "report",
                "GERAR RELATÓRIO",
                (40, 300, 360, 350),
                _ACCENT_BLUE,
                enabled=has_session,
            ),
            _Button("quit", "SAIR", (400, 300, 680, 350), _DISABLED),
        ]

    def _render(self) -> tuple:
        canvas = np.full((_H, _W, 3), _BG, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (_W, 70), _PANEL, -1)

        # Todo texto vai pelo renderizador Unicode (acentos corretos).
        items: list = [
            TextItem("Computing View — Presença & Uso de Celular",
                     org=(30, 22), font_px=24, color=_TEXT, bold=True),
        ]

        src = self.config.video_source_type
        src_txt = "webcam" if src == "webcam" else f"arquivo: {Path(self.config.video_path).name}"
        items.append(
            TextItem(f"Fonte de vídeo: {src_txt}", org=(40, 92), font_px=17, color=_MUTED)
        )

        buttons = self._buttons()
        for b in buttons:
            x1, y1, x2, y2 = b.box
            color = b.color if b.enabled else _DISABLED
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)
            txt_color = _TEXT if b.enabled else _MUTED
            items.append(
                TextItem(b.label, org=(x1 + 18, (y1 + y2) // 2 - 10),
                         font_px=18, color=txt_color)
            )

        # Rodapé: status + atalhos.
        cv2.rectangle(canvas, (0, _H - 80), (_W, _H), _PANEL, -1)
        items.append(TextItem(self._status[:78], org=(30, _H - 60), font_px=16, color=_ACCENT))
        items.append(
            TextItem("Atalhos: [I]niciar  [R]elatório  [Q]/ESC sair  |  na câmera: 'q' encerra",
                     org=(30, _H - 32), font_px=14, color=_MUTED)
        )

        default_renderer.render(canvas, items)
        return canvas, buttons

    # -- loop principal -----------------------------------------------------
    def run(self) -> int:
        cv2.namedWindow(self.WIN)
        cv2.setMouseCallback(self.WIN, self._on_mouse)
        try:
            while True:
                canvas, buttons = self._render()
                cv2.imshow(self.WIN, canvas)
                key = cv2.waitKey(30) & 0xFF

                action = None
                if self._click is not None:
                    cx, cy = self._click
                    self._click = None
                    for b in buttons:
                        if b.hit(cx, cy):
                            action = b.key
                            break
                if key in (ord("q"), 27):
                    action = "quit"
                elif key in (ord("i"), ord("I")):
                    action = "start"
                elif key in (ord("r"), ord("R")):
                    action = "report"
                elif key in (ord("m"), ord("M")):
                    action = "enroll"

                if action == "quit":
                    break
                if action == "enroll":
                    self._enroll()
                elif action == "start":
                    # Fecha o painel; o pipeline abre a própria janela de vídeo.
                    cv2.destroyWindow(self.WIN)
                    self._run_pipeline()
                    cv2.namedWindow(self.WIN)
                    cv2.setMouseCallback(self.WIN, self._on_mouse)
                elif action == "report":
                    if self._session is not None:
                        self._generate_report()
                    else:
                        self._status = "Rode uma sessão (INICIAR) antes de gerar o relatório."
        finally:
            cv2.destroyAllWindows()
        return 0

    # -- ações --------------------------------------------------------------
    def _enroll(self) -> None:
        self._status = "Matriculando alunos de data/students... (veja o terminal)"
        self._flush_once()
        try:
            from src.attendance.enrollment import enroll_directory
            from src.attendance.face_recognizer import FaceRecognizer

            cfg = self.config
            recognizer = FaceRecognizer(config=cfg)
            gallery, warnings = enroll_directory(
                cfg.students_dir,
                recognizer,
                gallery_path=cfg.gallery_path,
                enrolled_at=datetime.now().isoformat(timespec="seconds"),
                config=cfg,
            )
            msg = f"{len(gallery)} aluno(s) matriculado(s)."
            if warnings:
                msg += f" {len(warnings)} aviso(s) (veja o terminal)."
                for w in warnings:
                    print(f"[AVISO] {w}")
            self._status = msg
        except (RuntimeError, FileNotFoundError, ImportError) as exc:
            self._status = f"Erro na matrícula: {exc}"
            print(f"[ERRO] {exc}")

    def _build_config(self) -> Config:
        return replace(self.config, attendance_enabled=True)

    def _run_pipeline(self) -> None:
        from src.attendance.attendance import AttendanceTracker
        from src.attendance.enrollment import Gallery
        from src.attendance.face_recognizer import FaceRecognizer
        from src.detector import Detector
        from src.main import Application
        from src.temporal import TemporalTracker
        from src.video_source import VideoSourceFactory
        from src.visualizer import Visualizer

        cfg = self._build_config()
        self._status = "Carregando modelos e abrindo a câmera..."
        try:
            source = VideoSourceFactory.create(cfg)
            fps = getattr(source, "fps", float(cfg.output_fps))
            detector = Detector(config=cfg)
            visualizer = Visualizer(config=cfg)
            tracking = cfg.tracking_enabled and cfg.pose_enabled
            temporal = (
                TemporalTracker(cfg) if tracking and cfg.smoothing_enabled else None
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
            )
            app = Application(
                cfg, source, detector, visualizer,
                show_window=True,
                tracking=tracking,
                temporal=temporal,
                attendance_tracker=tracker,
            )
            print("[INFO] Câmera aberta. Pressione 'q' na janela do vídeo para encerrar.")
            app.run()
            self._session = tracker.finalize()
            self._status = "Sessão encerrada. Clique em GERAR RELATÓRIO."
        except (RuntimeError, FileNotFoundError) as exc:
            self._status = f"Erro: {exc}"
            print(f"[ERRO] {exc}")

    def _generate_report(self) -> None:
        from src.attendance.reports import generate_reports
        from src.attendance.ui import _open_folder

        self._status = "Gerando relatório..."
        self._flush_once()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = generate_reports(
            self._session, out_dir=self.config.reports_dir, stamp=stamp, config=self.config
        )
        folder = Path(paths[0]).parent if paths else Path(self.config.reports_dir)
        self._last_report_dir = folder
        self._status = f"Relatório salvo em {folder}"
        print(f"[INFO] Relatórios gerados em: {folder}")
        _open_folder(folder)

    def _flush_once(self) -> None:
        """Redesenha o painel uma vez para o status aparecer antes de bloquear."""
        canvas, _ = self._render()
        cv2.imshow(self.WIN, canvas)
        cv2.waitKey(1)


def run_cv_panel(config: Config | None = None) -> int:
    """Abre o painel OpenCV. Devolve 0 ao fechar."""
    return CvControlPanel(config or settings).run()
