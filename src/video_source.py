"""Fontes de vídeo — Padrão Strategy + Factory.

Objetivo: desacoplar a ORIGEM dos frames (webcam ao vivo vs. arquivo de
vídeo) da LÓGICA de detecção. O loop principal trabalha apenas contra a
interface abstrata ``VideoSource`` e não sabe (nem precisa saber) de onde
os frames vêm. Trocar webcam por arquivo é só mudar a configuração — o
código de detecção permanece intocado (Open/Closed + Dependency Inversion).

  - ``VideoSource``        : interface (Strategy abstrata).
  - ``WebcamSource``       : estratégia concreta para câmera ao vivo.
  - ``FileSource``         : estratégia concreta para arquivo de vídeo.
  - ``VideoSourceFactory`` : cria a estratégia certa a partir da config.

Todas as fontes implementam o protocolo de *context manager* (``with``),
garantindo que o recurso de captura seja sempre liberado.
"""

from __future__ import annotations

import abc
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, settings


class VideoSource(abc.ABC):
    """Interface (Strategy) para qualquer fonte de frames de vídeo."""

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    # -- contrato que cada estratégia concreta deve implementar ------------
    @abc.abstractmethod
    def _open(self) -> cv2.VideoCapture:
        """Cria e devolve o objeto cv2.VideoCapture já aberto."""
        raise NotImplementedError

    # -- comportamento comum a todas as fontes -----------------------------
    def open(self) -> None:
        """Abre a captura, validando que o recurso está acessível."""
        if self._capture is not None:
            return
        capture = self._open()
        if not capture or not capture.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo: {self!r}"
            )
        self._capture = capture

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Lê o próximo frame. Retorna (ok, frame) — espelha cv2.read()."""
        if self._capture is None:
            self.open()
        assert self._capture is not None  # para o type-checker
        ok, frame = self._capture.read()
        return ok, frame if ok else None

    def frames(self):
        """Gerador que itera sobre todos os frames até a fonte se esgotar."""
        self.open()
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield frame

    def release(self) -> None:
        """Libera o recurso de captura (idempotente)."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # -- suporte a "with VideoSource() as src:" ----------------------------
    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class WebcamSource(VideoSource):
    """Estratégia concreta: captura ao vivo de uma webcam local."""

    def __init__(self, device_index: int = 0) -> None:
        super().__init__()
        self.device_index = device_index

    def _open(self) -> cv2.VideoCapture:
        # Índice 0 = câmera padrão do sistema.
        return cv2.VideoCapture(self.device_index)

    def __repr__(self) -> str:
        return f"WebcamSource(device_index={self.device_index})"


class FileSource(VideoSource):
    """Estratégia concreta: lê frames de um arquivo de vídeo em disco."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    def _open(self) -> cv2.VideoCapture:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Arquivo de vídeo não encontrado: {self.path}"
            )
        return cv2.VideoCapture(str(self.path))

    def __repr__(self) -> str:
        return f"FileSource(path={self.path!s})"


class VideoSourceFactory:
    """Factory: decide qual estratégia instanciar a partir da configuração.

    Mantém o ponto de criação centralizado. Adicionar uma nova fonte (ex.:
    stream RTSP) é só criar a classe e registrar aqui, sem tocar no resto.
    """

    @staticmethod
    def create(config: Config | None = None) -> VideoSource:
        cfg = config or settings
        source_type = (cfg.video_source_type or "webcam").lower()

        if source_type == "webcam":
            return WebcamSource(device_index=cfg.webcam_index)
        if source_type == "file":
            return FileSource(path=cfg.video_path)

        raise ValueError(
            f"Tipo de fonte de vídeo desconhecido: '{cfg.video_source_type}'. "
            "Use 'webcam' ou 'file'."
        )
