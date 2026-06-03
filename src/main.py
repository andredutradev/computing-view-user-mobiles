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
        try:
            with self.video_source as source:
                for frame in source.frames():
                    # 1) Detecta + aplica regra de negócio.
                    people = self.detector.process_frame(frame)
                    # 2) Separa celulares para também desenhá-los.
                    phones = [
                        p.matched_phone
                        for p in people
                        if p.matched_phone is not None
                    ]
                    # 3) Renderiza.
                    annotated = self.visualizer.draw(frame, people, phones)

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

    return replace(settings, **overrides) if overrides else settings


def main(argv: list[str] | None = None) -> int:
    """Função main: monta as dependências e dispara a aplicação."""
    args = parse_args(argv)
    config = config_from_args(args)
    visualizer = Visualizer(config=config)

    if args.demo:
        # Modo demo: fonte sintética + detector roteirizado. Abre a janela
        # imediatamente, sem baixar pesos nem precisar de câmera/vídeo.
        from src.demo import ScriptedDetector, SyntheticVideoSource

        video_source: VideoSource = SyntheticVideoSource(
            max_frames=args.max_frames
        )
        detector: Detector = ScriptedDetector(config=config)
    else:
        video_source = VideoSourceFactory.create(config)
        detector = Detector(config=config)

    app = Application(
        config,
        video_source,
        detector,
        visualizer,
        show_window=not args.no_display,
        save_path=args.save,
    )
    try:
        app.run()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
