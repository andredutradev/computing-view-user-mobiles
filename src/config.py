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


def _env_int(name: str, default: int) -> int:
    """Lê uma variável de ambiente como inteiro, com fallback seguro."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    """Configuração imutável (frozen) da aplicação.

    Usar ``frozen=True`` evita que partes do sistema alterem a configuração
    em tempo de execução, tornando o comportamento previsível e testável.
    """

    # ------------------------------------------------------------------
    # Modelo YOLO
    # ------------------------------------------------------------------
    # Nome (ou caminho) do peso pré-treinado no COCO. Padrão "yolo11m.pt"
    # (modelo "medium" do YOLO11): em testes com celulares escuros/borrados na
    # mão, o "small"/"nano" PERDIAM o aparelho mesmo com confiança baixíssima,
    # enquanto o "medium" o detecta (~0.3). É o que sustenta a precisão pedida.
    # Para máquinas fracas / tempo real máximo, exporte CVUM_MODEL_PATH=yolov8n.pt.
    model_path: str = os.environ.get("CVUM_MODEL_PATH", "yolo11m.pt")

    # Modelo de POSE (esqueleto) usado para localizar braços e mãos das
    # pessoas. O modelo de pose detecta apenas a classe "person" e devolve
    # 17 keypoints COCO (ombros, cotovelos, PULSOS, etc.). O "small" do YOLO11
    # dá pulsos mais precisos (melhor a regra pulso↔celular). Para tempo real
    # máximo, exporte CVUM_POSE_MODEL=yolov8n-pose.pt.
    pose_model_path: str = os.environ.get("CVUM_POSE_MODEL", "yolo11s-pose.pt")

    # Liga/desliga o pipeline de pose. Com pose LIGADA (padrão), a decisão de
    # "segurando celular" usa a proximidade do PULSO ao celular — bem mais
    # preciso. Com pose DESLIGADA (CVUM_POSE=0), cai no caminho antigo de
    # modelo único + contenção (mais leve, para máquinas fracas).
    pose_enabled: bool = _env_bool("CVUM_POSE", True)

    # Confiança mínima para aceitar uma detecção (0.0 - 1.0). Usada para a
    # PESSOA/pose (objeto grande, alta confiança).
    confidence_threshold: float = _env_float("CVUM_CONF", 0.45)

    # Confiança mínima específica do CELULAR. Como o aparelho é pequeno, escuro
    # e muitas vezes parcialmente oculto na mão, ele costuma sair com confiança
    # mais baixa que a pessoa — por isso um limiar próprio, mais permissivo,
    # melhora bastante o recall sem afrouxar a detecção de pessoas.
    phone_confidence_threshold: float = _env_float("CVUM_PHONE_CONF", 0.25)

    # IoU usado pelo NMS interno do YOLO (não confundir com o IoU
    # pessoa↔celular calculado na regra de negócio).
    nms_iou_threshold: float = _env_float("CVUM_NMS_IOU", 0.45)

    # Tamanho da imagem (lado) para a inferência do YOLO. Subir para 960/1280
    # melhora MUITO o recall de objetos pequenos (o celular!), ao custo de mais
    # processamento — recomendado quando há GPU/MPS. 640 é o padrão equilibrado.
    imgsz: int = _env_int("CVUM_IMGSZ", 960)

    # ------------------------------------------------------------------
    # Dispositivo de inferência (adaptativo)
    # ------------------------------------------------------------------
    # "auto" escolhe o melhor disponível na ordem cuda -> mps (Apple) -> cpu.
    # Também aceita um valor explícito: "cpu", "cuda", "mps" ou "0" (índice GPU).
    device: str = os.environ.get("CVUM_DEVICE", "auto")

    # ------------------------------------------------------------------
    # Rastreamento (tracking) entre frames — IDs estáveis por pessoa
    # ------------------------------------------------------------------
    # Liga o rastreamento com ByteTrack/BoT-SORT (embutidos no ultralytics).
    # Dá um track_id estável a cada pessoa, base para a suavização temporal e
    # para o sistema de presença (dwell time por aluno).
    tracking_enabled: bool = _env_bool("CVUM_TRACK", True)
    # "bytetrack.yaml" (leve, ideal CPU) ou "botsort.yaml" (ReID, recupera
    # melhor de oclusões longas, porém mais pesado).
    tracker_config: str = os.environ.get("CVUM_TRACKER", "bytetrack.yaml")

    # ------------------------------------------------------------------
    # Suavização temporal / histerese do estado "usando celular"
    # ------------------------------------------------------------------
    # Resolve o problema do box piscando (laranja↔verde) com a movimentação:
    # em vez de decidir por frame isolado, votamos numa janela deslizante e
    # aplicamos histerese (limiares distintos para LIGAR e DESLIGAR) + um
    # "grace period" que mantém o estado durante oclusões breves.
    smoothing_enabled: bool = _env_bool("CVUM_SMOOTH", True)
    # Tamanho da janela de votos (frames). 15 ≈ 0,6s a 25fps.
    smoothing_window: int = _env_int("CVUM_SMOOTH_WINDOW", 15)
    # Fração de votos positivos necessária para LIGAR o estado (mais exigente).
    on_threshold: float = _env_float("CVUM_ON_THRESH", 0.6)
    # Fração abaixo da qual o estado DESLIGA (histerese: off < on).
    off_threshold: float = _env_float("CVUM_OFF_THRESH", 0.35)
    # Frames que um track sobrevive sem ser visto, mantendo seu último estado
    # (evita o box sumir em oclusões/movimentos rápidos).
    grace_period: int = _env_int("CVUM_GRACE", 10)
    # Frames até esquecer um track por completo (limita o uso de memória).
    track_max_age: int = _env_int("CVUM_TRACK_MAX_AGE", 30)

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
    # Regra de negócio: POSTURA de uso de celular (independe da foto/COCO)
    # ------------------------------------------------------------------
    # Em vez de depender SÓ de o YOLO ver o aparelho (que falha com celular
    # escuro/oculto na mão), inferimos a postura típica de "mexendo no celular"
    # a partir da POSE: mão erguida à frente do tronco + cotovelo flexionado +
    # cabeça inclinada para baixo (olhando a tela). Generaliza para casos não
    # presentes nas poucas fotos de calibração.
    posture_enabled: bool = _env_bool("CVUM_POSTURE", True)
    # Score [0..1] mínimo para, junto de uma caixa de celular próxima (ainda que
    # fraca), CONFIRMAR o uso (melhora o recall em celulares escuros).
    posture_assist_threshold: float = _env_float("CVUM_POSTURE_ASSIST", 0.45)
    # Score [0..1] mínimo para marcar uso APENAS pela postura (sem caixa de
    # celular). Mais alto, pois é o sinal mais sujeito a falso-positivo; a
    # suavização temporal/histerese ainda filtra oscilações.
    posture_standalone_threshold: float = _env_float("CVUM_POSTURE_SOLO", 0.70)
    # Fator de folga adicionado ao raio pulso↔celular quando a postura é forte
    # (a mão na posição típica "puxa" celulares um pouco mais distantes).
    posture_radius_bonus: float = _env_float("CVUM_POSTURE_RADIUS_BONUS", 0.6)

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

    # ------------------------------------------------------------------
    # Sistema de presença em sala (reconhecimento facial + dwell time)
    # ------------------------------------------------------------------
    # Liga o subsistema de presença. DESLIGADO por padrão: a detecção de
    # celular (e o --demo/tests) seguem idênticos quando não é pedido.
    attendance_enabled: bool = _env_bool("CVUM_ATTENDANCE", False)

    # InsightFace: pacote de modelos e limiares de reconhecimento.
    face_model_pack: str = os.environ.get("CVUM_FACE_PACK", "buffalo_l")
    # Usa GPU (CUDAExecutionProvider) automaticamente se o onnxruntime expor.
    face_use_gpu_if_available: bool = _env_bool("CVUM_FACE_GPU", True)
    # Tamanho de detecção do InsightFace (det_size). Maior = melhor com rostos
    # pequenos (sala ampla), porém mais lento.
    face_det_size: int = _env_int("CVUM_FACE_DET_SIZE", 640)
    # Similaridade de cosseno mínima (buffalo_l) para aceitar uma identidade.
    face_match_threshold: float = _env_float("CVUM_FACE_THRESH", 0.35)
    # Margem mínima entre o melhor e o 2º melhor ALUNO para aceitar a melhor
    # hipótese (anti-ambiguidade: dois alunos parecidos). 0.0 desliga.
    face_match_margin: float = _env_float("CVUM_FACE_MARGIN", 0.05)
    # Test-Time Augmentation no reconhecimento: embute o rosto E seu espelho
    # horizontal e usa a MÉDIA dos embeddings — mais estável a leves variações
    # de pose. Barato (1 passada extra só quando vamos reconhecer).
    face_tta_flip: bool = _env_bool("CVUM_FACE_TTA", True)
    # Matrícula: gera embeddings extra por foto (espelho horizontal) p/ robustez
    # a partir de UMA foto. Duplicatas (cosseno ~1) são descartadas.
    face_enroll_augment: bool = _env_bool("CVUM_FACE_ENROLL_AUG", True)
    # Score de detecção mínimo do InsightFace para ACEITAR um rosto na matrícula
    # (filtra fotos ruins/borradas que poluiriam a galeria).
    face_enroll_min_det_score: float = _env_float("CVUM_FACE_ENROLL_MIN_DET", 0.5)
    # Acima deste score, confirmamos a identidade de imediato (trava o cache).
    face_confirm_score: float = _env_float("CVUM_FACE_CONFIRM", 0.50)
    # Nº de acertos coerentes para confirmar a identidade de um track.
    face_confirm_hits: int = _env_int("CVUM_FACE_CONFIRM_HITS", 3)
    # Reconhecimento roda só a cada N frames por track ainda NÃO confirmado.
    # É o que mantém o tempo real em CPU (passada cara do InsightFace amortizada).
    face_recog_every_n_frames: int = _env_int("CVUM_FACE_EVERY_N", 10)
    # Tentativas máximas antes de desistir de identificar um track (fica "?").
    face_max_attempts: int = _env_int("CVUM_FACE_MAX_ATTEMPTS", 30)
    # Lado mínimo (px) de um rosto para tentar reconhecer (filtra ruído).
    face_min_size_px: int = _env_int("CVUM_FACE_MIN_PX", 40)
    # Fator de padding ao recortar a cabeça a partir dos keypoints faciais.
    face_roi_pad: float = _env_float("CVUM_FACE_ROI_PAD", 0.6)

    # Movimentação: velocidade do centroide normalizada pela altura da caixa.
    move_speed_threshold: float = _env_float("CVUM_MOVE_SPEED", 0.06)
    seated_speed_threshold: float = _env_float("CVUM_SEATED_SPEED", 0.02)
    seated_min_frames: int = _env_int("CVUM_SEATED_MIN_FRAMES", 15)

    # Presença/dwell: tolerância para sumiços breves (ponte) e tempo mínimo
    # para considerar o aluno "presente" no relatório.
    presence_grace_seconds: float = _env_float("CVUM_GRACE_SECONDS", 3.0)
    attendance_min_seconds: float = _env_float("CVUM_ATTEND_MIN_SECONDS", 30.0)

    # Caminhos (fotos de referência, galeria de embeddings e relatórios).
    students_dir: str = os.environ.get(
        "CVUM_STUDENTS_DIR", str(DATA_DIR / "students")
    )
    gallery_path: str = os.environ.get(
        "CVUM_GALLERY", str(DATA_DIR / "students" / "gallery.npz")
    )
    reports_dir: str = os.environ.get(
        "CVUM_REPORTS_DIR", str(DATA_DIR / "reports")
    )

    # Cores extra (BGR) para o sistema de presença.
    color_identified: tuple[int, int, int] = (0, 200, 200)  # amarelo: aluno reconhecido
    color_unknown: tuple[int, int, int] = (130, 130, 130)    # cinza: não identificado

    # Nome da janela de exibição
    window_name: str = "Computing View — Uso de Celular"

    # Mapa id->nome apenas para rótulos legíveis
    class_names: dict[int, str] = field(
        default_factory=lambda: {0: "Pessoa", 67: "Celular"}
    )


# Instância única reutilizável (importe `settings` onde precisar).
settings = Config()
