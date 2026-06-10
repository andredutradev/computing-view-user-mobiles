"""Harness de validação da detecção de celular em IMAGENS estáticas.

Roda o ``Detector`` (mesma regra pulso↔celular/contenção usada em vídeo, só que
por frame único, sem tracking/suavização) sobre uma pasta de imagens e reporta,
para cada uma: quantos celulares e pessoas foram detectados e se ALGUMA pessoa
ficou marcada como "usando celular".

Como todas as imagens da pasta de validação são exemplos POSITIVOS (pessoas que
estão usando o celular), o acerto é ``using_phone=True``. A taxa de acerto guia a
calibração dos limiares em ``config.py`` (confiança do celular, raio da mão,
contenção, etc.) sem precisar abrir a câmera.

Uso:
    python -m src.main --eval-images training_true_examples
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config import Config, settings
from src.detector import Detector

# Extensões de imagem aceitas na pasta de validação.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageResult:
    """Resultado da avaliação de UMA imagem."""

    path: Path
    people: int
    phones: int
    using_phone: bool
    # Maior confiança entre os celulares detectados (0.0 se nenhum).
    best_phone_conf: float = 0.0
    # Maior score de POSTURA entre as pessoas (sinal independente da caixa COCO).
    best_posture: float = 0.0
    # True se ALGUM acerto veio só da postura (sem caixa de celular associada).
    by_posture: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Acerto: a imagem é um positivo, então esperamos ``using_phone``."""
        return self.using_phone and self.error is None


def list_images(folder: Path) -> list[Path]:
    """Lista (ordenadas) as imagens suportadas dentro de ``folder``."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def evaluate_image(detector: Detector, path: Path, config: Config) -> ImageResult:
    """Avalia uma única imagem com o detector já carregado."""
    frame = cv2.imread(str(path))
    if frame is None:
        return ImageResult(path, 0, 0, False, error="não foi possível ler a imagem")

    people = detector.process_frame(frame)
    phones = [p.matched_phone for p in people if p.matched_phone is not None]
    using = any(p.using_phone for p in people)
    best_conf = max((ph.confidence for ph in phones), default=0.0)
    best_posture = max((p.posture_score for p in people), default=0.0)
    by_posture = any(p.using_phone and p.by_posture for p in people)
    return ImageResult(
        path=path,
        people=len(people),
        phones=len(phones),
        using_phone=using,
        best_phone_conf=best_conf,
        best_posture=best_posture,
        by_posture=by_posture,
    )


def evaluate_folder(
    folder: str | Path,
    config: Config | None = None,
    detector: Detector | None = None,
) -> list[ImageResult]:
    """Avalia todas as imagens da pasta. Carrega o YOLO uma única vez.

    ``detector`` pode ser injetado (testes com mocks); caso contrário um
    ``Detector`` real é construído a partir de ``config``.
    """
    cfg = config or settings
    base = Path(folder)
    if not base.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {base}")

    images = list_images(base)
    det = detector or Detector(config=cfg)
    det.load()  # carrega os pesos uma vez (no-op se já carregado/injetado)

    return [evaluate_image(det, path, cfg) for path in images]


def format_report(results: list[ImageResult]) -> str:
    """Monta a tabela legível + o resumo de taxa de acerto."""
    lines = [
        "Validação de detecção de celular (imagens = positivos)",
        "-" * 72,
        f"{'imagem':<38} {'pessoas':>7} {'cel':>4} {'conf':>6} "
        f"{'postura':>7} {'usando?':>8}",
        "-" * 80,
    ]
    for r in results:
        if r.error is not None:
            lines.append(f"{r.path.name:<38} ERRO: {r.error}")
            continue
        mark = "SIM" if r.using_phone else "NAO"
        if r.using_phone and r.by_posture:
            mark = "SIM*"  # * = confirmado pela postura (sem caixa de celular)
        lines.append(
            f"{r.path.name:<38} {r.people:>7} {r.phones:>4} "
            f"{r.best_phone_conf:>6.2f} {r.best_posture:>7.2f} {mark:>8}"
        )
    hits = sum(1 for r in results if r.ok)
    total = len(results)
    rate = (hits / total * 100.0) if total else 0.0
    posture_hits = sum(1 for r in results if r.ok and r.by_posture)
    lines.append("-" * 80)
    lines.append(f"Acertos (usando celular): {hits}/{total}  ({rate:.0f}%)")
    if posture_hits:
        lines.append(
            f"  dos quais {posture_hits} confirmados pela POSTURA (SIM*), "
            "sem caixa de celular detectada."
        )
    return "\n".join(lines)


def run(folder: str | Path, config: Config | None = None) -> int:
    """Entry point para o CLI (``--eval-images``). Imprime o relatório.

    Devolve 0 se todas as imagens foram reconhecidas como uso de celular,
    1 caso contrário (útil para uso em scripts/CI de calibração).
    """
    results = evaluate_folder(folder, config)
    print(format_report(results))
    if not results:
        print("[AVISO] Nenhuma imagem encontrada na pasta.")
        return 1
    return 0 if all(r.ok for r in results) else 1
