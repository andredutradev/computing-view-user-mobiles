"""Painel de controle moderno em **PySide6 (Qt)** — interface principal.

Por que Qt (e não Tkinter): o Tk 8.5 do macOS abre a janela preta (problema já
documentado no projeto) e não renderiza acentos de forma confiável. O Qt renderiza
texto Unicode nativamente (acentos do português corretos), tem visual moderno e
embute o vídeo direto na janela — então NÃO usamos ``cv2.imshow`` (que no macOS
exige a main thread e brigaria com o loop do Qt).

Arquitetura de threads:
  - A UI vive na main thread (Qt).
  - O pipeline de monitoramento (``Application``) roda numa ``QThread`` com
    ``show_window=False``; cada frame anotado volta para a UI por um **signal**
    (thread-safe) que pinta um ``QLabel``. Um ``threading.Event`` encerra o loop.
  - O preview da webcam na matrícula usa um ``QTimer`` que lê frames na própria
    main thread (sem thread extra) — liberado antes de iniciar o monitoramento
    para não disputar a câmera.

Fluxo de matrícula pedido: campo de **nome** + botão **Capturar foto** (tira a
foto da webcam ao vivo, salva em ``data/students/<slug>/`` e matricula na hora).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, settings


def _np_to_qpixmap(frame_bgr, target_size):
    """Converte um frame BGR (numpy) em QPixmap escalado, mantendo proporção."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if target_size is not None and target_size.width() > 0:
        pix = pix.scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pix


def _build_pipeline_worker_cls():
    """Cria a classe do worker (import do Qt adiado para dentro da função)."""
    from PySide6.QtCore import QThread, Signal

    class PipelineWorker(QThread):
        """Roda o ``Application`` (monitoramento) fora da main thread."""

        frameReady = Signal(object)   # frame anotado (numpy BGR)
        statusMsg = Signal(str)
        finishedSession = Signal(object)  # Session ou None

        def __init__(self, cfg: Config, stop_event, recognizer=None):
            super().__init__()
            self._cfg = cfg
            self._stop = stop_event
            self._recognizer = recognizer

        def run(self):  # noqa: D401 - método do QThread
            try:
                from src.attendance.attendance import AttendanceTracker
                from src.attendance.enrollment import Gallery
                from src.attendance.face_recognizer import FaceRecognizer
                from src.detector import Detector
                from src.main import Application
                from src.temporal import TemporalTracker
                from src.video_source import VideoSourceFactory
                from src.visualizer import Visualizer

                cfg = self._cfg
                self.statusMsg.emit("Carregando modelos e abrindo a câmera...")
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
                recognizer = self._recognizer or FaceRecognizer(config=cfg)
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
                self.statusMsg.emit("Monitorando... (clique em Parar para encerrar)")

                def on_frame(annotated, people, idx):
                    self.frameReady.emit(annotated)

                app = Application(
                    cfg, source, detector, visualizer,
                    show_window=False,           # vídeo embutido na UI, sem cv2.imshow
                    tracking=tracking,
                    temporal=temporal,
                    attendance_tracker=tracker,
                    stop_event=self._stop,
                    on_frame=on_frame,
                )
                app.run()
                self.finishedSession.emit(tracker.finalize())
            except Exception as exc:  # erro visível na barra de status
                self.statusMsg.emit(f"Erro: {exc}")
                self.finishedSession.emit(None)

    return PipelineWorker


class QtControlPanel:
    """Janela principal do sistema (PySide6)."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or settings
        self._stop_event = threading.Event()
        self._worker = None
        self._session = None
        self._recognizer = None
        self._preview_cap = None
        self._last_report_dir: Path | None = None

    # -- recognizer compartilhado (lazy) -----------------------------------
    def _get_recognizer(self):
        if self._recognizer is None:
            from src.attendance.face_recognizer import FaceRecognizer

            self._recognizer = FaceRecognizer(config=self.config)
        return self._recognizer

    # -- construção da UI ---------------------------------------------------
    def run(self) -> int:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )

        self._Qt = Qt
        self._QFileDialog = QFileDialog

        app = QApplication.instance() or QApplication([])
        app.setStyleSheet(_STYLE)

        win = QWidget()
        win.setWindowTitle("Computing View — Presença & Uso de Celular")
        win.resize(1040, 680)
        self._win = win

        root = QHBoxLayout(win)

        # ---- Coluna esquerda: vídeo ----
        left = QVBoxLayout()
        title = QLabel("Computing View")
        title.setObjectName("title")
        title.setFont(QFont("", 20, QFont.Bold))
        left.addWidget(title)
        subtitle = QLabel("Detecção de uso de celular + presença por reconhecimento facial")
        subtitle.setObjectName("subtitle")
        left.addWidget(subtitle)

        self.video_label = QLabel("A câmera aparece aqui.")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setFrameShape(QFrame.StyledPanel)
        left.addWidget(self.video_label, stretch=1)
        root.addLayout(left, stretch=3)

        # ---- Coluna direita: controles ----
        right = QVBoxLayout()
        right.setSpacing(12)

        # (1) Matrícula
        gb_enroll = QGroupBox("1 · Matrícula de alunos")
        ge = QGridLayout(gb_enroll)
        ge.addWidget(QLabel("Nome do aluno:"), 0, 0, 1, 2)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Ex.: André Conceição")
        ge.addWidget(self.name_edit, 1, 0, 1, 2)
        self.preview_btn = QPushButton("Ligar webcam")
        self.preview_btn.clicked.connect(self._toggle_preview)
        ge.addWidget(self.preview_btn, 2, 0)
        self.capture_btn = QPushButton("📷  Capturar e matricular")
        self.capture_btn.setObjectName("accent")
        self.capture_btn.clicked.connect(self._capture_and_enroll)
        ge.addWidget(self.capture_btn, 2, 1)
        self.enroll_dir_btn = QPushButton("Matricular pasta data/students")
        self.enroll_dir_btn.clicked.connect(self._enroll_folder)
        ge.addWidget(self.enroll_dir_btn, 3, 0, 1, 2)
        right.addWidget(gb_enroll)

        # (2) Fonte de vídeo + monitoramento
        gb_run = QGroupBox("2 · Monitoramento")
        gr = QGridLayout(gb_run)
        gr.addWidget(QLabel("Fonte:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "Arquivo de vídeo"])
        self.source_combo.setCurrentIndex(
            1 if self.config.video_source_type == "file" else 0
        )
        gr.addWidget(self.source_combo, 0, 1)
        self.video_path_edit = QLineEdit(self.config.video_path)
        gr.addWidget(self.video_path_edit, 1, 0, 1, 1)
        browse = QPushButton("Procurar…")
        browse.clicked.connect(self._pick_video)
        gr.addWidget(browse, 1, 1)
        gr.addWidget(QLabel("Sensibilidade do reconhecimento:"), 2, 0, 1, 2)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(20, 70)
        self.thresh_slider.setValue(int(self.config.face_match_threshold * 100))
        self.thresh_label = QLabel(f"{self.config.face_match_threshold:.2f}")
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_label.setText(f"{v / 100:.2f}")
        )
        gr.addWidget(self.thresh_slider, 3, 0)
        gr.addWidget(self.thresh_label, 3, 1)
        self.start_btn = QPushButton("▶  Iniciar")
        self.start_btn.setObjectName("accent")
        self.start_btn.clicked.connect(self._start)
        gr.addWidget(self.start_btn, 4, 0)
        self.stop_btn = QPushButton("■  Parar")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        gr.addWidget(self.stop_btn, 4, 1)
        right.addWidget(gb_run)

        # (3) Relatório
        gb_rep = QGroupBox("3 · Relatório")
        gp = QVBoxLayout(gb_rep)
        self.report_btn = QPushButton("Gerar relatório (CSV + PDF)")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self._generate_report)
        gp.addWidget(self.report_btn)
        right.addWidget(gb_rep)

        right.addStretch(1)
        self.status_label = QLabel("Pronto.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)
        root.addLayout(right, stretch=2)

        # Timer do preview da webcam (matrícula).
        self._preview_timer = QTimer(win)
        self._preview_timer.timeout.connect(self._update_preview)

        win.closeEvent = self._on_close  # type: ignore[assignment]
        win.show()
        return app.exec()

    # -- preview da webcam (matrícula) -------------------------------------
    def _toggle_preview(self) -> None:
        if self._preview_timer.isActive():
            self._stop_preview()
            self.preview_btn.setText("Ligar webcam")
            self.video_label.setText("Preview desligado.")
        else:
            cap = cv2.VideoCapture(int(self.config.webcam_index))
            if not cap.isOpened():
                self._set_status("Não consegui abrir a webcam.")
                cap.release()
                return
            self._preview_cap = cap
            self._preview_timer.start(33)  # ~30 fps
            self.preview_btn.setText("Desligar webcam")
            self._set_status("Webcam ligada. Enquadre o rosto e capture.")

    def _stop_preview(self) -> None:
        if self._preview_timer.isActive():
            self._preview_timer.stop()
        if self._preview_cap is not None:
            self._preview_cap.release()
            self._preview_cap = None

    def _update_preview(self) -> None:
        if self._preview_cap is None:
            return
        ok, frame = self._preview_cap.read()
        if not ok:
            return
        self._last_preview_frame = frame
        self.video_label.setPixmap(
            _np_to_qpixmap(frame, self.video_label.size())
        )

    def _capture_and_enroll(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self._set_status("Digite o nome do aluno antes de capturar.")
            return
        frame = getattr(self, "_last_preview_frame", None)
        if frame is None:
            self._set_status("Ligue a webcam e aguarde o preview antes de capturar.")
            return
        self._set_status("Processando rosto e matriculando...")
        self._win.repaint()
        from src.attendance.enrollment import enroll_capture

        ok, msg, _ = enroll_capture(
            name,
            frame.copy(),
            self._get_recognizer(),
            cfg=self.config,
            enrolled_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._set_status(msg)
        if ok:
            self.name_edit.clear()

    def _enroll_folder(self) -> None:
        self._set_status("Matriculando alunos de data/students...")
        self._win.repaint()
        from src.attendance.enrollment import enroll_directory

        gallery, warnings = enroll_directory(
            self.config.students_dir,
            self._get_recognizer(),
            gallery_path=self.config.gallery_path,
            enrolled_at=datetime.now().isoformat(timespec="seconds"),
            config=self.config,
        )
        msg = f"{len(gallery)} aluno(s) na galeria."
        if warnings:
            msg += f" {len(warnings)} aviso(s) (veja o terminal)."
            for w in warnings:
                print(f"[AVISO] {w}")
        self._set_status(msg)

    # -- monitoramento ------------------------------------------------------
    def _pick_video(self) -> None:
        f, _ = self._QFileDialog.getOpenFileName(
            self._win, "Escolha um vídeo", "",
            "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos (*.*)",
        )
        if f:
            self.video_path_edit.setText(f)
            self.source_combo.setCurrentIndex(1)

    def _build_config(self) -> Config:
        return replace(
            self.config,
            attendance_enabled=True,
            face_match_threshold=self.thresh_slider.value() / 100.0,
            video_source_type="file" if self.source_combo.currentIndex() == 1 else "webcam",
            video_path=self.video_path_edit.text(),
        )

    def _start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._stop_preview()  # libera a câmera do preview antes do pipeline
        self.preview_btn.setText("Ligar webcam")
        self._stop_event.clear()
        self._session = None
        cfg = self._build_config()
        Worker = _build_pipeline_worker_cls()
        self._worker = Worker(cfg, self._stop_event, recognizer=self._get_recognizer())
        self._worker.frameReady.connect(self._on_pipeline_frame)
        self._worker.statusMsg.connect(self._set_status)
        self._worker.finishedSession.connect(self._on_pipeline_done)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.report_btn.setEnabled(False)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self._set_status("Parando...")

    def _on_pipeline_frame(self, frame) -> None:
        self.video_label.setPixmap(_np_to_qpixmap(frame, self.video_label.size()))

    def _on_pipeline_done(self, session) -> None:
        self._session = session
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if session is not None and getattr(session, "students", None):
            self.report_btn.setEnabled(True)
            self._set_status("Sessão encerrada. Você já pode gerar o relatório.")

    # -- relatório ----------------------------------------------------------
    def _generate_report(self) -> None:
        if self._session is None:
            self._set_status("Nenhuma sessão para relatar.")
            return
        self._set_status("Gerando relatório...")
        self._win.repaint()
        from src.attendance.reports import generate_reports
        from src.attendance.ui import _open_folder

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = generate_reports(
            self._session, out_dir=self.config.reports_dir, stamp=stamp, config=self.config
        )
        folder = Path(paths[0]).parent if paths else Path(self.config.reports_dir)
        self._last_report_dir = folder
        self._set_status(f"Relatório salvo em {folder}")
        _open_folder(folder)

    # -- utilidades ---------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_close(self, event) -> None:
        self._stop_event.set()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        self._stop_preview()
        event.accept()


# Folha de estilo (QSS) — visual escuro moderno, acentos renderizados nativamente.
_STYLE = """
QWidget { background: #1e1f24; color: #e8e8ea; font-size: 14px; }
QLabel#title { color: #ffffff; }
QLabel#subtitle { color: #9aa0aa; font-size: 13px; margin-bottom: 6px; }
QLabel#video { background: #111216; border-radius: 8px; color: #6b7280; }
QLabel#status { background: #15161a; border: 1px solid #2a2c33; border-radius: 6px;
                padding: 8px; color: #8fd19e; }
QGroupBox { border: 1px solid #2a2c33; border-radius: 8px; margin-top: 10px;
            padding: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #c9ccd3; }
QLineEdit, QComboBox { background: #26282f; border: 1px solid #34373f;
                       border-radius: 6px; padding: 6px; }
QPushButton { background: #2d2f37; border: 1px solid #3a3d46; border-radius: 6px;
              padding: 8px 10px; }
QPushButton:hover { background: #363943; }
QPushButton:disabled { color: #6b7280; background: #24252b; }
QPushButton#accent { background: #2e7d32; border: none; color: white; font-weight: 600; }
QPushButton#accent:hover { background: #388e3c; }
QSlider::groove:horizontal { height: 6px; background: #34373f; border-radius: 3px; }
QSlider::handle:horizontal { background: #2e7d32; width: 16px; margin: -6px 0;
                             border-radius: 8px; }
"""


def run_qt_panel(config: Config | None = None) -> int:
    """Abre o painel PySide6. Devolve 0 ao fechar.

    Levanta ``ImportError`` se o PySide6 não estiver instalado — o ``run_ui``
    captura isso e cai para um backend alternativo.
    """
    import PySide6  # noqa: F401 - valida a presença antes de construir a janela

    return QtControlPanel(config or settings).run()
