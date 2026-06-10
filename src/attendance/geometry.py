"""Geometria de apoio ao reconhecimento facial.

Duas funções centrais:
  - ``head_roi_from_keypoints``: estima a caixa da CABEÇA a partir dos keypoints
    faciais da pose (nariz/olhos/orelhas). É bem mais justa que a caixa inteira
    da pessoa — essencial para reconhecer rostos pequenos em planos abertos de
    sala. Quando não há keypoints faciais confiáveis, cai para o topo da caixa.
  - ``crop_box``: recorta com segurança uma região do frame (com clamp).

Tudo aqui é puro/numpy — fácil de testar sem InsightFace.
"""

from __future__ import annotations

import numpy as np

from src.config import Config

# Índices COCO dos keypoints faciais (mesma convenção do detector de pose).
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
FACE_KEYPOINTS = (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR)

Box = "tuple[float, float, float, float]"


def _top_region(person_box, frac: float = 0.35):
    """Fallback: o topo (cabeça/ombros) da caixa da pessoa."""
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + frac * (y2 - y1))


def head_roi_from_keypoints(keypoints, person_box, cfg: Config):
    """Caixa aproximada da cabeça (x1,y1,x2,y2) para recortar o rosto.

    Usa os keypoints faciais confiáveis (conf >= ``wrist_conf_threshold``);
    aplica padding proporcional ao tamanho do rosto. Sem keypoints faciais
    utilizáveis, devolve o topo da caixa da pessoa.
    """
    if keypoints is not None:
        pts = [
            (float(keypoints[i][0]), float(keypoints[i][1]))
            for i in FACE_KEYPOINTS
            if i < len(keypoints) and keypoints[i][2] >= cfg.wrist_conf_threshold
        ]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            spread = max(maxx - minx, maxy - miny)
            person_w = max(1.0, person_box[2] - person_box[0])
            # Com poucos pontos (ex.: só o nariz) o spread ~0: usa fração da
            # largura da pessoa como tamanho base do rosto.
            size = spread if spread > 1.0 else 0.25 * person_w
            padx = cfg.face_roi_pad * size
            pady = cfg.face_roi_pad * size
            # Mais folga para cima (testa/cabelo) e para baixo (queixo).
            return (
                minx - padx,
                miny - pady * 1.5,
                maxx + padx,
                maxy + pady * 1.5,
            )
    return _top_region(person_box)


def clamp_box(box, width: int, height: int):
    """Converte para int e prende a caixa nos limites do frame."""
    x1, y1, x2, y2 = box
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


def crop_box(frame: np.ndarray, box):
    """Recorta ``box`` de ``frame`` com segurança. Devolve None se vazio."""
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normaliza L2 um vetor de embedding (evita divisão por zero)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-8:
        return v
    return v / norm


def cosine_match(embedding: np.ndarray, matrix: np.ndarray, ids: list):
    """Melhor correspondência por cosseno entre ``embedding`` e ``matrix``.

    Assume ``matrix`` (N, D) e ``embedding`` (D,) já normalizados L2; o produto
    interno equivale ao cosseno. Devolve ``(id_vencedor, score)`` ou
    ``(None, 0.0)`` se a galeria estiver vazia.
    """
    if matrix is None or len(matrix) == 0 or not ids:
        return (None, 0.0)
    emb = normalize(embedding)
    sims = matrix @ emb
    best = int(np.argmax(sims))
    return (ids[best], float(sims[best]))
