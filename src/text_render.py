"""Renderização de texto com acentuação correta sobre frames do OpenCV.

O ``cv2.putText`` usa as fontes Hershey (vetoriais, ASCII apenas) — acentos do
português (á, ã, ç, ê, õ…) saem como "?" ou caixas. Para resolver isso de uma
vez, este módulo desenha o texto com **Pillow** (FreeType/TrueType, Unicode
completo) sobre o frame BGR do OpenCV.

Decisões de design:
  - **Uma fonte empacotada** (``assets/fonts/DejaVuSans.ttf``) garante o mesmo
    resultado em qualquer máquina/CI; há uma cadeia de fallback para fontes do
    sistema e, em último caso, a fonte default do Pillow.
  - ``ImageFont`` é **cacheado por (caminho, tamanho)** — abrir a fonte a cada
    frame seria caro.
  - A API é **em lote** (``render``): converte o frame BGR→RGB **uma única vez**
    por frame, desenha todos os rótulos e converte de volta. Desenhar texto por
    caixa convertendo o frame inteiro a cada chamada seria O(frame × pessoas).

As cores entram no padrão **BGR** (como no resto do projeto) e são convertidas
para RGB internamente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:  # Pillow é dependência do projeto; o import protegido evita quebra dura.
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except Exception:  # pragma: no cover - caminho defensivo
    _PIL_OK = False

_BASE_DIR = Path(__file__).resolve().parent.parent
_BUNDLED_FONT = _BASE_DIR / "assets" / "fonts" / "DejaVuSans.ttf"
_BUNDLED_FONT_BOLD = _BASE_DIR / "assets" / "fonts" / "DejaVuSans-Bold.ttf"

# Fontes do sistema com cobertura latina completa, por plataforma.
_SYSTEM_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",       # macOS
    "/Library/Fonts/Arial.ttf",                            # macOS (legado)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",              # Fedora
    "C:\\Windows\\Fonts\\arial.ttf",                       # Windows
)


def _resolve_font_path(bold: bool = False) -> str | None:
    """Resolve o caminho da fonte TrueType, na ordem de preferência.

    1) ``CVUM_FONT_PATH`` (override explícito);
    2) a fonte empacotada no repositório (determinística);
    3) fontes comuns do sistema;
    4) ``None`` -> o chamador usa a fonte default do Pillow.
    """
    env = os.environ.get("CVUM_FONT_PATH")
    if env and Path(env).exists():
        return env
    bundled = _BUNDLED_FONT_BOLD if bold else _BUNDLED_FONT
    if bundled.exists():
        return str(bundled)
    if _BUNDLED_FONT.exists():  # fallback: regular se não houver bold
        return str(_BUNDLED_FONT)
    for cand in _SYSTEM_FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    return None


@lru_cache(maxsize=64)
def _load_font(size: int, bold: bool):
    """Carrega (e cacheia) a ``ImageFont`` no tamanho pedido."""
    path = _resolve_font_path(bold=bold)
    if path is not None:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:  # pragma: no cover - fonte corrompida/indisponível
            pass
    # Último recurso: fonte bitmap embutida do Pillow (sem tamanho real, mas
    # nunca quebra). Acentos básicos do latin-1 ainda saem corretos.
    return ImageFont.load_default()


def _bgr_to_rgb(color: tuple) -> tuple:
    """(B, G, R) -> (R, G, B), com clamp em [0, 255] e inteiros."""
    b, g, r = (int(max(0, min(255, c))) for c in color[:3])
    return (r, g, b)


@dataclass
class TextItem:
    """Um rótulo a desenhar: texto, posição (canto sup-esq), estilo.

    ``bg`` (BGR) opcional desenha uma caixa de fundo atrás do texto, para
    legibilidade sobre qualquer cena (mesma ideia do código antigo em cv2).
    """

    text: str
    org: tuple  # (x, y) do canto superior-esquerdo do texto
    font_px: int = 18
    color: tuple = (255, 255, 255)  # BGR
    bg: tuple | None = None         # BGR ou None
    pad: int = 3
    bold: bool = False


class TextRenderer:
    """Desenha texto Unicode (acentos OK) sobre frames BGR do OpenCV."""

    def __init__(self) -> None:
        self.available = _PIL_OK

    # -- medição ------------------------------------------------------------
    def measure(self, text: str, font_px: int = 18, bold: bool = False) -> tuple:
        """Retorna ``(largura, altura)`` em pixels do texto renderizado."""
        if not self.available:
            return (len(text) * font_px // 2, font_px)
        font = _load_font(int(font_px), bold)
        # textbbox em (0,0) dá a caixa do texto; usamos largura/altura dela.
        l, t, r, b = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            (0, 0), text, font=font
        )
        return (r - l, b - t)

    # -- desenho em lote ----------------------------------------------------
    def render(self, img_bgr: np.ndarray, items: list) -> np.ndarray:
        """Desenha todos os ``items`` no frame, com UMA conversão BGR↔RGB.

        Retorna o MESMO array (modificado in-place via cópia de pixels) para
        ser compatível com o estilo do Visualizer; se o Pillow não estiver
        disponível, cai para ``cv2.putText`` (sem acentos, mas não quebra).
        """
        if not items:
            return img_bgr
        if not self.available:  # pragma: no cover - Pillow ausente
            self._fallback_cv2(img_bgr, items)
            return img_bgr

        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for it in items:
            font = _load_font(int(it.font_px), it.bold)
            x, y = int(it.org[0]), int(it.org[1])
            if it.bg is not None:
                l, t, r, b = draw.textbbox((x, y), it.text, font=font)
                draw.rectangle(
                    (l - it.pad, t - it.pad, r + it.pad, b + it.pad),
                    fill=_bgr_to_rgb(it.bg),
                )
            draw.text((x, y), it.text, font=font, fill=_bgr_to_rgb(it.color))

        # Copia de volta para o array original (preserva a referência/contrato).
        np.copyto(img_bgr, cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
        return img_bgr

    def _fallback_cv2(self, img_bgr: np.ndarray, items: list) -> None:
        """Sem Pillow: desenha com cv2 (perde acentos, mas funciona)."""
        for it in items:
            scale = max(0.4, it.font_px / 30.0)
            x, y = int(it.org[0]), int(it.org[1] + it.font_px)
            if it.bg is not None:
                (tw, th), bl = cv2.getTextSize(
                    it.text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
                )
                cv2.rectangle(
                    img_bgr, (x - it.pad, y - th - it.pad),
                    (x + tw + it.pad, y + bl), it.bg, cv2.FILLED
                )
            cv2.putText(
                img_bgr, it.text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, it.color, 1, cv2.LINE_AA,
            )


# Instância única reutilizável (o cache de fontes é global/processo).
default_renderer = TextRenderer()
