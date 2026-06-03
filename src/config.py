"""Configurações globais da aplicação.

Centraliza todos os parâmetros ajustáveis (caminhos, limiares, classes,
cores e fonte de vídeo) em um único ponto. Isso atende ao princípio de
*Single Responsibility* do SOLID: nenhuma outra classe precisa conhecer
valores "mágicos" espalhados pelo código — todos vivem aqui.

Os valores podem ser sobrescritos por variáveis de ambiente, o que é útil
para rodar em ambientes diferentes (CI, produção, máquina local) sem editar
o código-fonte.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Diretório raiz do projeto (… / computing-view-user-mobiles)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _env_float(name: str, default: float) -> float:
    """Lê uma variável de ambiente como float, com fallback seguro."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Lê uma variável de ambiente como booleano (0/1, true/false, no/yes)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class Config:
    """Configuração imutável (frozen) da aplicação.

    Usar ``frozen=True`` evita que partes do sistema alterem a configuração
    em tempo de execução, tornando o comportamento previsível e testável.
    """

    # ------------------------------------------------------------------
    # Modelo YOLO
    # ------------------------------------------------------------------
    # Nome (ou caminho) do peso pré-treinado no COCO. "yolov8n.pt" é o
    # modelo "nano" — leve e rápido, ideal para tempo real. Para mais
    # precisão troque por "yolov8s.pt", "yol11n.pt", etc.
    model_path: str = os.environ.get("CVUM_MODEL_PATH", "yolov8n.pt")

    # Modelo de POSE (esqueleto) usado para localizar braços e mãos das
    # pessoas. O modelo de pose detecta apenas a classe "person" e devolve
    # 17 keypoints COCO (ombros, cotovelos, PULSOS, etc.). É leve ("nano")
    # e roda junto com o modelo de detecção acima (que cuida do celular).
    pose_model_path: str = os.environ.get("CVUM_POSE_MODEL", "yolov8n-pose.pt")

    # Liga/desliga o pipeline de pose. Com pose LIGADA (padrão), a decisão de
    # "segurando celular" usa a proximidade do PULSO ao celular — bem mais
    # preciso. Com pose DESLIGADA (CVUM_POSE=0), cai no caminho antigo de
    # modelo único + contenção (mais leve, para máquinas fracas).
    pose_enabled: bool = _env_bool("CVUM_POSE", True)

    # Confiança mínima para aceitar uma detecção (0.0 - 1.0).
    confidence_threshold: float = _env_float("CVUM_CONF", 0.45)

    # IoU usado pelo NMS interno do YOLO (não confundir com o IoU
    # pessoa↔celular calculado na regra de negócio).
    nms_iou_threshold: float = _env_float("CVUM_NMS_IOU", 0.45)

    # ------------------------------------------------------------------
    # Classes do COCO que nos interessam
    # ------------------------------------------------------------------
    # IDs oficiais no dataset COCO: 0 = person, 67 = cell phone.
    person_class_id: int = 0
    phone_class_id: int = 67

    # ------------------------------------------------------------------
    # Regra de negócio: associação pessoa ↔ celular
    # ------------------------------------------------------------------
    # IoU mínimo entre a caixa da pessoa e a do celular para considerar
    # "uso". Como o celular costuma ser bem menor que a pessoa, o IoU puro
    # tende a ser baixo; por isso usamos também a "contenção" (fração do
    # celular dentro da pessoa) — ver detector.py.
    association_iou_threshold: float = _env_float("CVUM_ASSOC_IOU", 0.01)

    # Fração mínima da área do celular que precisa estar dentro da caixa
    # da pessoa para marcá-la como "usando" (0.0 - 1.0). Mais robusto que
    # o IoU para objetos de tamanhos muito diferentes. Usado como FALLBACK
    # quando os pulsos da pessoa não estão visíveis/confiáveis.
    containment_threshold: float = _env_float("CVUM_CONTAINMENT", 0.50)

    # ------------------------------------------------------------------
    # Regra de negócio: proximidade do PULSO (mão) ao celular
    # ------------------------------------------------------------------
    # Confiança mínima de um keypoint de pulso para confiarmos nele. Abaixo
    # disso o pulso é considerado "não visível" e usamos a contenção.
    wrist_conf_threshold: float = _env_float("CVUM_WRIST_CONF", 0.30)

    # Raio de proximidade pulso↔celular, como FRAÇÃO de uma referência de
    # escala da pessoa (largura dos ombros; na falta, largura da caixa). Se o
    # pulso está a uma distância <= raio da caixa do celular, consideramos
    # que a pessoa está segurando o aparelho. Escala-invariante (perto/longe
    # da câmera). Aumente para ficar mais permissivo.
    hand_radius_factor: float = _env_float("CVUM_HAND_RADIUS", 0.5)

    # ------------------------------------------------------------------
    # Cores (BGR — padrão do OpenCV) e estilo de desenho
    # ------------------------------------------------------------------
    # OBS.: cores no padrão BGR do OpenCV. Por requisito do projeto, o box
    # da pessoa fica VERDE quando ela está usando celular (destaque do alvo)
    # e LARANJA quando não está. O celular em si é desenhado em azul.
    color_idle: tuple[int, int, int] = (0, 140, 255)       # laranja: sem celular
    color_using_phone: tuple[int, int, int] = (0, 220, 0)  # verde: usando celular
    color_phone: tuple[int, int, int] = (255, 180, 0)      # azul: celular
    color_target: tuple[int, int, int] = (0, 255, 0)       # verde-alvo (crosshair)
    color_text: tuple[int, int, int] = (255, 255, 255)     # branco: texto

    # Esqueleto (pose): braços e mãos.
    color_arm: tuple[int, int, int] = (255, 255, 0)        # ciano: braços (ombro→cotovelo→pulso)
    color_hand: tuple[int, int, int] = (255, 0, 255)       # magenta: mãos (pulsos)
    color_hand_active: tuple[int, int, int] = (0, 0, 255)  # vermelho: mão que segura o celular

    box_thickness: int = 2
    font_scale: float = 0.6

    # FPS usado ao gravar a saída anotada em arquivo (--save).
    output_fps: int = 25

    # ------------------------------------------------------------------
    # Fonte de vídeo (padrão Strategy/Factory — ver video_source.py)
    # ------------------------------------------------------------------
    # "webcam" -> cv2.VideoCapture(0); "file" -> usa video_path.
    video_source_type: str = os.environ.get("CVUM_SOURCE", "webcam")
    webcam_index: int = int(os.environ.get("CVUM_WEBCAM_INDEX", "0"))
    video_path: str = os.environ.get(
        "CVUM_VIDEO_PATH", str(DATA_DIR / "sample_video.mp4")
    )

    # Nome da janela de exibição
    window_name: str = "Computing View — Uso de Celular"

    # Mapa id->nome apenas para rótulos legíveis
    class_names: dict[int, str] = field(
        default_factory=lambda: {0: "Pessoa", 67: "Celular"}
    )


# Instância única reutilizável (importe `settings` onde precisar).
settings = Config()
