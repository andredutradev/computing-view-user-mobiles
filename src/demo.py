"""Modo DEMO — abre a interface sem precisar de YOLO, câmera ou vídeo.

Por que existe: permite "abrir a interface mesmo em teste". Em vez de rodar
o modelo real (que exige baixar pesos e ter webcam/arquivo), geramos uma
CENA SINTÉTICA determinística — uma pessoa e um celular que se aproxima,
sobrepõe e se afasta. A detecção é "roteirizada" (scripted), mas a regra de
negócio que decide "usando celular" é EXATAMENTE a mesma de produção
(``Detector.associate`` + geometria de ``detector.py``).

Assim você vê o box da pessoa alternar para VERDE (com o alvo) quando o
celular a sobrepõe, validando a lógica e a interface de ponta a ponta.

Componentes (mantêm os mesmos contratos das versões reais):
  - ``demo_scene(index)``      : função pura -> (person_box, phone_box|None).
  - ``SyntheticVideoSource``   : VideoSource que "renderiza" a cena.
  - ``ScriptedDetector``       : Detector cujo YOLO é substituído pelo roteiro.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.config import Config, settings
from src.detector import (
    KP_L_ELBOW,
    KP_L_SHOULDER,
    KP_L_WRIST,
    KP_R_ELBOW,
    KP_R_SHOULDER,
    KP_R_WRIST,
    ARM_SEGMENTS,
    Detection,
    Detector,
    PersonDetection,
)
from src.video_source import VideoSource

# Caixa = (x1, y1, x2, y2)
Box = tuple[float, float, float, float]

# Dimensões do "palco" sintético (altura, largura).
FRAME_H, FRAME_W = 480, 640
# Duração de um ciclo completo da animação, em frames.
CYCLE = 160


def _demo_keypoints(person: Box, phone: Box | None) -> np.ndarray:
    """Gera keypoints COCO (17, 3) sintéticos para a pessoa.

    Só preenchemos ombros/cotovelos/pulsos (braços e mãos) — o resto fica com
    confiança 0. Quando há celular, o BRAÇO DIREITO se estende até ele (o pulso
    pousa sobre o celular), disparando a MESMA regra de pulso-proximidade de
    produção. Sem celular, os braços ficam ao lado do corpo.
    """
    x1, y1, x2, y2 = person
    cx = (x1 + x2) / 2.0
    sh_y = y1 + 90.0  # altura dos ombros

    kp = np.zeros((17, 3), dtype=float)
    kp[KP_L_SHOULDER] = (cx - 35, sh_y, 0.95)
    kp[KP_R_SHOULDER] = (cx + 35, sh_y, 0.95)
    # Braços padrão (descansando ao lado do corpo).
    kp[KP_L_ELBOW] = (cx - 48, sh_y + 80, 0.90)
    kp[KP_L_WRIST] = (cx - 52, sh_y + 160, 0.90)
    kp[KP_R_ELBOW] = (cx + 48, sh_y + 80, 0.90)
    kp[KP_R_WRIST] = (cx + 52, sh_y + 160, 0.90)

    if phone is not None:
        px = (phone[0] + phone[2]) / 2.0
        py = (phone[1] + phone[3]) / 2.0
        # Braço direito alcança o celular: cotovelo no meio do caminho,
        # pulso (mão) exatamente sobre o aparelho.
        kp[KP_R_ELBOW] = ((cx + 35 + px) / 2.0, (sh_y + py) / 2.0, 0.92)
        kp[KP_R_WRIST] = (px, py, 0.93)

    return kp


def demo_scene(index: int) -> tuple[Box, np.ndarray, Box | None]:
    """Define, de forma determinística, a cena no frame ``index``.

    Retorna a caixa da pessoa, seus keypoints (braços/mãos) e (opcionalmente)
    a caixa do celular. O celular só existe num trecho do ciclo, durante o qual
    ele cruza a pessoa; nesse trecho a mão direita o segura.
    """
    phase = index % CYCLE

    # Pessoa: balança levemente na horizontal (sway) usando seno do índice.
    cx = int(FRAME_W / 2 + 30 * math.sin(index / 18.0))
    person: Box = (cx - 60, 80, cx + 60, 440)

    # Celular: presente entre os frames 40 e 120 do ciclo, deslizando da
    # esquerda para a direita. No meio do trajeto ele passa SOBRE a pessoa.
    phone: Box | None = None
    if 40 <= phase < 120:
        px = 120 + (phase - 40) * 4  # de x=120 até ~x=440
        py = 250
        phone = (px - 22, py - 38, px + 22, py + 38)

    keypoints = _demo_keypoints(person, phone)
    return person, keypoints, phone


def _draw_person_silhouette(
    frame: np.ndarray, box: Box, keypoints: np.ndarray
) -> None:
    """Desenha uma silhueta simples (corpo + cabeça + braços) para visual."""
    x1, y1, x2, y2 = (int(v) for v in box)
    body_color = (90, 90, 90)
    cv2.rectangle(frame, (x1, y1 + 40), (x2, y2), body_color, cv2.FILLED)
    head_r = (x2 - x1) // 3
    cx = (x1 + x2) // 2
    cv2.circle(frame, (cx, y1 + 40), head_r, body_color, cv2.FILLED)

    # Braços (ombro→cotovelo→pulso) desenhados como "membros" da silhueta.
    for a, b in ARM_SEGMENTS:
        pa = (int(keypoints[a][0]), int(keypoints[a][1]))
        pb = (int(keypoints[b][0]), int(keypoints[b][1]))
        cv2.line(frame, pa, pb, body_color, 14, cv2.LINE_AA)


def _draw_phone_icon(frame: np.ndarray, box: Box) -> None:
    """Desenha um retângulo claro representando o celular."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), cv2.FILLED)
    cv2.rectangle(frame, (x1 + 4, y1 + 6, x2 - x1 - 8, y2 - y1 - 16),
                  (40, 40, 40), 1)


class SyntheticVideoSource(VideoSource):
    """Fonte de vídeo que gera frames sintéticos a partir de ``demo_scene``.

    Implementa o mesmo contrato de ``VideoSource`` (open/read/frames/release),
    então pluga no loop principal sem qualquer alteração na orquestração.
    """

    def __init__(self, max_frames: int | None = None) -> None:
        super().__init__()
        self.max_frames = max_frames  # None = roda indefinidamente
        self._index = 0
        self._opened = False

    def _open(self):  # pragma: no cover - não usamos cv2.VideoCapture aqui
        raise NotImplementedError("Fonte sintética não usa cv2.VideoCapture.")

    def open(self) -> None:
        self._opened = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.max_frames is not None and self._index >= self.max_frames:
            return False, None

        person, keypoints, phone = demo_scene(self._index)
        # Fundo escuro uniforme.
        frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
        _draw_person_silhouette(frame, person, keypoints)
        if phone is not None:
            _draw_phone_icon(frame, phone)

        self._index += 1
        return True, frame

    def release(self) -> None:
        self._opened = False

    def __repr__(self) -> str:
        return f"SyntheticVideoSource(max_frames={self.max_frames})"


class ScriptedDetector(Detector):
    """Detector cujo "YOLO" é o roteiro de ``demo_scene``.

    Sobrescreve ``detect_people``/``detect_phones``: em vez de rodar as redes
    neurais, devolve as pessoas (com keypoints) e os celulares roteirizados.
    Toda a regra de negócio (``associate`` + pulso-proximidade/contenção) é
    herdada e executada tal como em produção — é isso que faz o box ficar verde
    no momento certo, agora pela proximidade da MÃO ao celular.
    """

    def __init__(self, config: Config | None = None) -> None:
        # Passa modelos "fake" não-nulos para evitar qualquer carga de peso.
        super().__init__(
            config=config or settings, model=object(), pose_model=object()
        )
        self._index = 0

    def load(self) -> None:  # noop: não há modelo real para carregar
        return None

    def detect_people(self, frame: np.ndarray) -> list[PersonDetection]:
        self._validate_frame(frame)
        person, keypoints, _ = demo_scene(self._index)
        return [
            PersonDetection(
                self.config.person_class_id, 0.98, person, keypoints=keypoints
            )
        ]

    def detect_phones(self, frame: np.ndarray) -> list[Detection]:
        self._validate_frame(frame)
        _, _, phone = demo_scene(self._index)
        if phone is None:
            return []
        return [Detection(self.config.phone_class_id, 0.92, phone)]

    def process_frame(self, frame: np.ndarray) -> list[PersonDetection]:
        # Usa o índice corrente para pessoa E celular, e só então avança —
        # garante que ambos descrevem o MESMO frame da animação.
        people = self.detect_people(frame)
        phones = self.detect_phones(frame)
        self._index += 1
        return self.associate(people, phones)
