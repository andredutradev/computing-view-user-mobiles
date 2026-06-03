"""Camada de visualização — desenha caixas e rótulos sobre o frame.

Responsabilidade única: dado um frame e a lista de pessoas já processadas
(``PersonDetection``), produzir um frame anotado. Não roda inferência nem
conhece o YOLO — recebe os resultados prontos. Isso mantém a renderização
totalmente desacoplada da detecção (SRP), e permite trocar o estilo visual
sem mexer na lógica.

Convenção de cor (BGR, padrão OpenCV):
  - Laranja  : pessoa SEM celular.
  - Verde    : pessoa USANDO celular (+ alvo/crosshair).
  - Azul     : caixa do próprio celular.
  - Ciano    : braços (ombro→cotovelo→pulso).
  - Magenta  : mãos (pulsos); a mão que segura o celular fica em vermelho.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import Config, settings
from src.detector import (
    ARM_SEGMENTS,
    KP_L_WRIST,
    KP_R_WRIST,
    Detection,
    PersonDetection,
)


class Visualizer:
    """Desenha bounding boxes e rótulos legíveis sobre os frames."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or settings
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    # -- API pública --------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        people: list[PersonDetection],
        phones: list[Detection] | None = None,
    ) -> np.ndarray:
        """Anota o frame com as pessoas (e opcionalmente os celulares).

        Trabalha sobre uma cópia para não mutar o frame original — útil se
        o chamador quiser preservar o frame cru (ex.: gravar em disco).
        """
        annotated = frame.copy()
        cfg = self.config

        # Celulares primeiro, para que as caixas de pessoa fiquem por cima.
        for phone in phones or []:
            self._draw_box(
                annotated,
                phone.box,
                color=cfg.color_phone,
                label=f"{cfg.class_names.get(phone.class_id, 'Celular')} "
                f"{phone.confidence:.2f}",
            )

        for person in people:
            using = person.using_phone
            color = cfg.color_using_phone if using else cfg.color_idle
            status = "Usando Celular" if using else "Pessoa"
            label = f"{status} {person.confidence:.2f}"
            self._draw_box(annotated, person.box, color=color, label=label)
            # Esqueleto: braços (ombro→cotovelo→pulso) e mãos (pulsos). Ajuda
            # a enxergar POR QUE a pessoa foi (ou não) marcada como usando.
            self._draw_skeleton(annotated, person)
            # Para quem está usando celular, reforça o "alvo": cantos em L
            # e um crosshair no centro, deixando o box destacado em verde.
            if using:
                self._draw_target(annotated, person.box)

        self._draw_hud(annotated, people)
        return annotated

    # -- helpers internos ---------------------------------------------------
    def _draw_box(
        self,
        img: np.ndarray,
        box: tuple[float, float, float, float],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Desenha um retângulo + rótulo com fundo preenchido."""
        cfg = self.config
        x1, y1, x2, y2 = (int(round(v)) for v in box)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, cfg.box_thickness)

        # Caixa de fundo do texto para legibilidade sobre qualquer cena.
        (tw, th), baseline = cv2.getTextSize(
            label, self._font, cfg.font_scale, 1
        )
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(
            img, (x1, top), (x1 + tw + 4, y1), color, thickness=cv2.FILLED
        )
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - baseline - 2),
            self._font,
            cfg.font_scale,
            cfg.color_text,
            1,
            cv2.LINE_AA,
        )

    def _draw_skeleton(self, img: np.ndarray, person: PersonDetection) -> None:
        """Desenha os braços (linhas) e as mãos (círculos nos pulsos).

        Só desenha pontos/segmentos com confiança >= limiar. A mão que está
        segurando o celular (``person.holding_wrist``) sai destacada.
        """
        kp = person.keypoints
        if kp is None:
            return
        cfg = self.config
        conf_t = cfg.wrist_conf_threshold

        # Braços: ombro→cotovelo→pulso de cada lado.
        for a, b in ARM_SEGMENTS:
            if kp[a][2] >= conf_t and kp[b][2] >= conf_t:
                pa = (int(round(kp[a][0])), int(round(kp[a][1])))
                pb = (int(round(kp[b][0])), int(round(kp[b][1])))
                cv2.line(img, pa, pb, cfg.color_arm, cfg.box_thickness, cv2.LINE_AA)

        # Mãos: círculo em cada pulso confiável; destaca o que segura o celular.
        held = person.holding_wrist
        for w in (KP_L_WRIST, KP_R_WRIST):
            if kp[w][2] < conf_t:
                continue
            wx, wy = int(round(kp[w][0])), int(round(kp[w][1]))
            active = (
                held is not None
                and abs(wx - held[0]) <= 1.5
                and abs(wy - held[1]) <= 1.5
            )
            color = cfg.color_hand_active if active else cfg.color_hand
            radius = 9 if active else 6
            cv2.circle(img, (wx, wy), radius, color, -1, cv2.LINE_AA)
            if active:
                # Anel extra para reforçar a mão que segura.
                cv2.circle(img, (wx, wy), radius + 4, color, 2, cv2.LINE_AA)

    def _draw_target(
        self, img: np.ndarray, box: tuple[float, float, float, float]
    ) -> None:
        """Desenha um "alvo" sobre o box: cantos em L + crosshair central.

        Puramente estético — destaca visualmente a pessoa identificada como
        usando celular (o box já está verde; isto reforça o "target").
        """
        cfg = self.config
        color = cfg.color_target
        x1, y1, x2, y2 = (int(round(v)) for v in box)

        # Comprimento de cada "perna" do canto em L (proporcional ao box).
        leg = max(12, int(0.18 * min(x2 - x1, y2 - y1)))
        t = cfg.box_thickness + 1
        # Quatro cantos (superior-esq, superior-dir, inferior-esq, inf-dir).
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(img, (cx, cy), (cx + dx * leg, cy), color, t)
            cv2.line(img, (cx, cy), (cx, cy + dy * leg), color, t)

        # Crosshair no centro do box.
        ccx, ccy = (x1 + x2) // 2, (y1 + y2) // 2
        gap, arm = 4, 10
        cv2.line(img, (ccx - arm, ccy), (ccx - gap, ccy), color, 1, cv2.LINE_AA)
        cv2.line(img, (ccx + gap, ccy), (ccx + arm, ccy), color, 1, cv2.LINE_AA)
        cv2.line(img, (ccx, ccy - arm), (ccx, ccy - gap), color, 1, cv2.LINE_AA)
        cv2.line(img, (ccx, ccy + gap), (ccx, ccy + arm), color, 1, cv2.LINE_AA)
        cv2.circle(img, (ccx, ccy), 2, color, -1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, people: list[PersonDetection]) -> None:
        """Desenha um resumo (HUD) no topo do frame."""
        cfg = self.config
        total = len(people)
        using = sum(1 for p in people if p.using_phone)
        text = f"Pessoas: {total} | Usando celular: {using}"

        (tw, th), baseline = cv2.getTextSize(
            text, self._font, cfg.font_scale, 1
        )
        cv2.rectangle(img, (0, 0), (tw + 10, th + baseline + 8), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            img,
            text,
            (5, th + 4),
            self._font,
            cfg.font_scale,
            cfg.color_text,
            1,
            cv2.LINE_AA,
        )
