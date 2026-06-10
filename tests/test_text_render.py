"""Testes do renderizador de texto Unicode (acentos sobre frames OpenCV)."""

from __future__ import annotations

import numpy as np

from src.text_render import TextItem, TextRenderer, _bgr_to_rgb


def test_bgr_to_rgb_inverts_channels():
    assert _bgr_to_rgb((255, 128, 0)) == (0, 128, 255)
    # Clamp e int.
    assert _bgr_to_rgb((300.0, -5.0, 10.5)) == (10, 0, 255)


def test_measure_positive_dimensions():
    r = TextRenderer()
    w, h = r.measure("Atenção", font_px=20)
    assert w > 0 and h > 0


def test_render_empty_items_returns_same_array():
    r = TextRenderer()
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    out = r.render(img, [])
    assert out is img


def test_render_draws_pixels_with_accents():
    """Desenhar um texto acentuado deve alterar pixels (e não quebrar)."""
    r = TextRenderer()
    img = np.zeros((60, 240, 3), dtype=np.uint8)
    before = img.copy()
    out = r.render(
        img,
        [TextItem("Presença à noção", org=(5, 10), font_px=22,
                  color=(255, 255, 255), bg=(0, 0, 0))],
    )
    assert out.shape == before.shape
    # Algum pixel mudou (o texto/fundo foi desenhado).
    assert np.any(out != before)


def test_render_preserves_array_identity():
    """``render`` escreve in-place no mesmo array (contrato com o Visualizer)."""
    r = TextRenderer()
    img = np.zeros((50, 120, 3), dtype=np.uint8)
    out = r.render(img, [TextItem("olá", org=(2, 2), font_px=18)])
    assert out is img
