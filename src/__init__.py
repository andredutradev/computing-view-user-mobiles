"""Pacote principal do sistema de detecção de uso de celular.

Expõe as classes públicas mais usadas para facilitar a importação:

    from src import Detector, VideoSourceFactory, Visualizer
"""

from src.detector import Detector, Detection, PersonDetection
from src.video_source import (
    VideoSource,
    WebcamSource,
    FileSource,
    VideoSourceFactory,
)
from src.visualizer import Visualizer

__all__ = [
    "Detector",
    "Detection",
    "PersonDetection",
    "VideoSource",
    "WebcamSource",
    "FileSource",
    "VideoSourceFactory",
    "Visualizer",
]
