"""Ponto de entrada da aplicação — orquestra o pipeline em tempo real.

Junta as três camadas desacopladas:
    VideoSource (de onde vêm os frames)
        -> Detector (o que há no frame + regra de negócio)
            -> Visualizer (como desenhar o resultado)

O loop principal não conhece detalhes de nenhuma delas além de suas
interfaces — segue Dependency Inversion. Para trocar webcam por arquivo,
basta exportar CVUM_SOURCE=file (e CVUM_VIDEO_PATH=...) ou passar --source.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src.config import Config, settings
from src.detector import Detector
from src.temporal import TemporalTracker
from src.video_source import VideoSource, VideoSourceFactory
from src.visualizer import Visualizer


class Application:
    """Orquestrador do pipeline de detecção em tempo real."""

    def __init__(
        self,
        config: Config,
        video_source: VideoSource,
        detector: Detector,
        visualizer: Visualizer,
        show_window: bool = True,
        save_path: str | None = None,
        tracking: bool = False,
        temporal: TemporalTracker | None = None,
        attendance_tracker: object | None = None,
        stop_event: object | None = None,
        on_frame: object | None = None,
    ) -> None:
        # Todas as dependências são injetadas — facilita teste e troca.
        self.config = config
        self.video_source = video_source
        self.detector = detector
        self.visualizer = visualizer
        # show_window=False permite rodar sem GUI (CI/headless/validação).
        self.show_window = show_window
        # save_path != None grava a saída anotada em um arquivo de vídeo.
        self.save_path = save_path
        # tracking=True usa o caminho com rastreamento (track_id estável).
        self.tracking = tracking
        # temporal != None aplica suavização/histerese ao estado "usando".
        self.temporal = temporal
        # attendance_tracker != None alimenta o sistema de presença em sala.
        self.attendance_tracker = attendance_tracker
        # stop_event (threading.Event): permite a UI pedir parada limpa.
        self.stop_event = stop_event
        # on_frame(annotated, people, frame_index): callback opcional por frame
        # (a UI usa para atualizar a barra de status sem acoplar à janela cv2).
        self.on_frame = on_frame

    def run(self) -> None:
        """Executa o loop principal até o vídeo acabar ou o usuário sair."""
        cfg = self.config
        print(f"[INFO] Fonte de vídeo: {self.video_source!r}")
        print(f"[INFO] Modelo (detecção/celular): {cfg.model_path}")
        if cfg.pose_enabled:
            print(f"[INFO] Modelo (pose/braços e mãos): {cfg.pose_model_path}")
        else:
            print("[INFO] Pose DESLIGADA (fallback por contenção).")
        if self.save_path:
            print(f"[INFO] Gravando saída anotada em: {self.save_path}")
        if self.show_window:
            print("[INFO] Pressione 'q' ou ESC para sair.")

        self.detector.load()  # carrega o peso YOLO uma única vez

        writer: cv2.VideoWriter | None = None
        frame_index = 0
        try:
            with self.video_source as source:
                for frame in source.frames():
                    # 0) Parada solicitada pela UI (botão "Parar").
                    if self.stop_event is not None and self.stop_event.is_set():
                        break

                    # 1) Detecta + aplica regra de negócio (com ou sem tracking).
                    if self.tracking:
                        people = self.detector.process_frame_tracked(frame)
                    else:
                        people = self.detector.process_frame(frame)

                    # 1b) Suavização temporal (consistência do estado entre frames).
                    if self.temporal is not None:
                        people = self.temporal.update(people, frame_index)

                    # 1c) Sistema de presença em sala (opcional). Recebe o frame
                    #     cru para recortar rostos e reconhecer os alunos.
                    if self.attendance_tracker is not None:
                        self.attendance_tracker.update(frame, people, frame_index)

                    # 2) Separa celulares para também desenhá-los.
                    phones = [
                        p.matched_phone
                        for p in people
                        if p.matched_phone is not None
                    ]
                    # 3) Renderiza (com identidades dos alunos, se houver).
                    identities = (
                        self.attendance_tracker.identity_labels()
                        if self.attendance_tracker is not None
                        else None
                    )
                    annotated = self.visualizer.draw(
                        frame, people, phones, identities=identities
                    )

                    # 4) Grava em arquivo, se solicitado (writer lazy: só aqui
                    #    sabemos as dimensões reais do frame anotado).
                    if self.save_path:
                        if writer is None:
                            h, w = annotated.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(
                                self.save_path, fourcc, cfg.output_fps, (w, h)
                            )
                        writer.write(annotated)

                    # 5) Callback opcional por frame (a UI usa para status).
                    if self.on_frame is not None:
                        self.on_frame(annotated, people, frame_index)

                    frame_index += 1

                    if not self.show_window:
                        continue  # modo headless: processa sem abrir janela

                    cv2.imshow(cfg.window_name, annotated)
                    # waitKey(1) = ~tempo real; lê tecla de saída.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' ou ESC
                        break
        finally:
            if writer is not None:
                writer.release()
            if self.show_window:
                cv2.destroyAllWindows()
        print("[INFO] Encerrado.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Faz o parsing dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Detecção de uso de celular por pessoas em vídeo/webcam."
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "file"],
        help="Fonte de vídeo (sobrescreve CVUM_SOURCE).",
    )
    parser.add_argument(
        "--video", help="Caminho do arquivo de vídeo (quando --source=file)."
    )
    parser.add_argument("--model", help="Caminho/nome do peso YOLO.")
    parser.add_argument(
        "--conf", type=float, help="Limiar de confiança (0.0-1.0)."
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Desliga a pose (braços/mãos); usa só contenção (mais leve).",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Desliga o rastreamento entre frames (sem track_id estável).",
    )
    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "botsort"],
        help="Algoritmo de rastreamento (padrão: bytetrack).",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Desliga a suavização temporal/histerese (volta a piscar).",
    )
    parser.add_argument(
        "--laptop",
        action="store_true",
        help="Liga o reconhecimento de NOTEBOOK (classe COCO 63) para não marcar "
        "quem digita como 'usando celular'. Padrão: desligado (só celular, mais leve).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo de inferência (padrão: auto -> cuda/mps/cpu).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Lado da imagem na inferência (ex.: 960/1280 -> melhor recall do celular).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Abre o painel de controle desktop (Tkinter): upload de vídeo, "
        "matrícula e geração de relatórios.",
    )
    parser.add_argument(
        "--attendance",
        action="store_true",
        help="Liga o sistema de presença (reconhecimento facial + dwell time).",
    )
    parser.add_argument(
        "--enroll",
        metavar="PATH",
        default=None,
        help="Matricula os alunos a partir da pasta de fotos e encerra.",
    )
    parser.add_argument(
        "--eval-images",
        metavar="PATH",
        default=None,
        help="Valida a detecção de celular numa pasta de imagens e encerra "
        "(imprime acertos/total). Útil para calibrar limiares.",
    )
    parser.add_argument(
        "--report-out",
        metavar="DIR",
        default=None,
        help="Pasta de saída dos relatórios (padrão: data/reports).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Abre a interface com uma cena SINTÉTICA (sem YOLO/câmera).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Roda sem abrir janela (headless) — útil para CI/validação.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Encerra após N frames (principalmente para --demo/--no-display).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Grava a saída anotada em um arquivo de vídeo (ex.: out.mp4).",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    """Constrói a Config final aplicando os overrides de CLI sobre os defaults."""
    # dataclasses.replace cria uma nova Config imutável com overrides.
    from dataclasses import replace

    overrides: dict = {}
    if args.source:
        overrides["video_source_type"] = args.source
    if args.video:
        overrides["video_path"] = args.video
    if args.model:
        overrides["model_path"] = args.model
    if args.conf is not None:
        overrides["confidence_threshold"] = args.conf
    if args.no_pose:
        overrides["pose_enabled"] = False
    if args.no_track:
        overrides["tracking_enabled"] = False
    if args.tracker:
        overrides["tracker_config"] = f"{args.tracker}.yaml"
    if args.no_smooth:
        overrides["smoothing_enabled"] = False
    if args.laptop:
        overrides["laptop_suppression_enabled"] = True
    if args.device:
        overrides["device"] = args.device
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.attendance:
        overrides["attendance_enabled"] = True
    if args.report_out:
        overrides["reports_dir"] = args.report_out

    return replace(settings, **overrides) if overrides else settings


def _run_enrollment(config: Config, photos_dir: str) -> int:
    """Matrícula headless: calcula embeddings das fotos e salva a galeria."""
    from datetime import datetime

    from src.attendance.enrollment import enroll_directory
    from src.attendance.face_recognizer import FaceRecognizer

    recognizer = FaceRecognizer(config=config)
    gallery, warnings = enroll_directory(
        photos_dir,
        recognizer,
        gallery_path=config.gallery_path,
        enrolled_at=datetime.now().isoformat(timespec="seconds"),
        config=config,
    )
    print(f"[INFO] {len(gallery)} aluno(s) matriculado(s) -> {config.gallery_path}")
    for w in warnings:
        print(f"[AVISO] {w}")
    return 0


def _build_attendance_tracker(config: Config, video_source: VideoSource):
    """Monta o AttendanceTracker (galeria + reconhecedor) para a sessão."""
    from datetime import datetime

    from src.attendance.attendance import AttendanceTracker
    from src.attendance.enrollment import Gallery
    from src.attendance.face_recognizer import FaceRecognizer

    fps = getattr(video_source, "fps", float(config.output_fps))
    gallery = Gallery.load(config.gallery_path)
    recognizer = FaceRecognizer(config=config)
    label = (
        config.video_path
        if config.video_source_type == "file"
        else f"webcam_{config.webcam_index}"
    )
    return AttendanceTracker(
        config=config,
        gallery=gallery,
        recognizer=recognizer,
        fps=fps,
        source_label=label,
        started_at=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    )


def main(argv: list[str] | None = None) -> int:
    """Função main: monta as dependências e dispara a aplicação."""
    args = parse_args(argv)
    config = config_from_args(args)

    # Painel desktop (Tkinter): assume o controle e encerra ao fechar a janela.
    if args.ui:
        from src.attendance.ui import run_ui

        return run_ui(config)

    # Validação da detecção de celular numa pasta de imagens e sai.
    if args.eval_images:
        from src.eval_images import run as run_eval_images

        try:
            return run_eval_images(args.eval_images, config)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ERRO] {exc}", file=sys.stderr)
            return 1

    # Matrícula headless e sai.
    if args.enroll:
        try:
            return _run_enrollment(config, args.enroll)
        except (RuntimeError, FileNotFoundError, ImportError) as exc:
            print(f"[ERRO] {exc}", file=sys.stderr)
            return 1

    visualizer = Visualizer(config=config)

    if args.demo:
        # Modo demo: fonte sintética + detector roteirizado. Abre a janela
        # imediatamente, sem baixar pesos nem precisar de câmera/vídeo. O demo
        # NÃO usa tracking/suavização (cena roteirizada de uma única pessoa).
        from src.demo import ScriptedDetector, SyntheticVideoSource

        video_source: VideoSource = SyntheticVideoSource(
            max_frames=args.max_frames
        )
        detector: Detector = ScriptedDetector(config=config)
        tracking = False
        temporal = None
    else:
        video_source = VideoSourceFactory.create(config)
        detector = Detector(config=config)
        # Tracking só faz sentido com pose ligada (rastreamos pelo modelo de
        # pose). A suavização depende do tracking (precisa de track_id).
        tracking = config.tracking_enabled and config.pose_enabled
        temporal = (
            TemporalTracker(config)
            if tracking and config.smoothing_enabled
            else None
        )

    # Sistema de presença (opcional): só fora do demo e quando solicitado.
    attendance_tracker = None
    if config.attendance_enabled and not args.demo:
        attendance_tracker = _build_attendance_tracker(config, video_source)

    app = Application(
        config,
        video_source,
        detector,
        visualizer,
        show_window=not args.no_display,
        save_path=args.save,
        tracking=tracking,
        temporal=temporal,
        attendance_tracker=attendance_tracker,
    )
    try:
        app.run()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário.")
        return 0

    # Ao fim da sessão de presença, fecha os intervalos e gera os relatórios.
    if attendance_tracker is not None:
        from datetime import datetime

        from src.attendance.reports import generate_reports

        now = datetime.now()
        session = attendance_tracker.finalize(
            ended_at=now.strftime("%d/%m/%Y %H:%M:%S")
        )
        stamp = now.strftime("%Y%m%d_%H%M%S")
        paths = generate_reports(
            session, out_dir=config.reports_dir, stamp=stamp, config=config
        )
        if paths:
            print(f"[INFO] Relatórios gerados em: {paths[0].parent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
