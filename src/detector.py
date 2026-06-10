"""Camada de detecção (wrapper do YOLO) e regra de negócio.

Responsabilidades (Single Responsibility):
  1. Encapsular o modelo YOLO (carregar pesos, rodar inferência num frame).
  2. Filtrar apenas as classes de interesse (pessoa e celular).
  3. Aplicar a lógica de negócio: decidir, por geometria, quais pessoas
     estão "usando celular".

A camada NÃO desenha nada na imagem — isso é responsabilidade do
``Visualizer``. Assim, detecção e visualização ficam desacopladas e cada
uma pode ser testada/alterada isoladamente (Open/Closed + SRP).

------------------------------------------------------------------------
LÓGICA MATEMÁTICA DA SOBREPOSIÇÃO (explicada em detalhe)
------------------------------------------------------------------------
Cada *bounding box* é representado pelos cantos (x1, y1, x2, y2), onde
(x1, y1) é o canto superior-esquerdo e (x2, y2) o inferior-direito, em
coordenadas de pixel (eixo y cresce para baixo, padrão de imagem).

(1) Área de uma caixa A = (x2 - x1) * (y2 - y1).

(2) Interseção de duas caixas A e B:
    A região de sobreposição também é um retângulo. Seus cantos são:
        ix1 = max(Ax1, Bx1)   ix2 = min(Ax2, Bx2)
        iy1 = max(Ay1, By1)   iy2 = min(Ay2, By2)
    A largura/altura da interseção podem ser negativas (quando NÃO há
    sobreposição), então truncamos em zero:
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        area_inter = iw * ih

(3) IoU (Intersection over Union):
        IoU = area_inter / (area_A + area_B - area_inter)
    Varia de 0 (nenhuma sobreposição) a 1 (caixas idênticas). É a métrica
    clássica, porém penaliza objetos de tamanhos muito diferentes: um
    celular (caixa pequena) totalmente dentro de uma pessoa (caixa grande)
    produz IoU baixo, porque o denominador (a união) é dominado pela área
    da pessoa.

(4) Contenção (containment ratio) — métrica complementar:
        containment = area_inter / area_celular
    Responde: "que fração do celular está dentro da pessoa?". É robusta
    para a nossa tarefa, pois o celular é pequeno: se ~50%+ dele cai dentro
    da caixa da pessoa, é muito provável que ela o esteja segurando.

(5) Distância euclidiana entre centros (desempate/proximidade):
        centro = ((x1 + x2) / 2, (y1 + y2) / 2)
        dist = sqrt((cx_p - cx_c)^2 + (cy_p - cy_c)^2)
    Usada para, entre várias pessoas candidatas, atribuir o celular à
    pessoa cujo centro está mais próximo do centro do celular.

Decisão final: uma pessoa é marcada como "usando celular" se EXISTE um
celular tal que (IoU >= limiar_iou) OU (containment >= limiar_contenção).
Quando vários candidatos passam, o celular é atribuído ao mais próximo
(menor distância euclidiana) para evitar contagem dupla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import Config, settings


# ---------------------------------------------------------------------------
# Índices dos keypoints no padrão COCO (17 pontos), usados pelo modelo de pose.
# Só nos interessam os do tronco superior — ombro→cotovelo→PULSO — que formam
# os braços e localizam as mãos.
# ---------------------------------------------------------------------------
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
# Segmentos (ombro→cotovelo→pulso) de cada lado, para desenhar os braços.
ARM_SEGMENTS = (
    (KP_L_SHOULDER, KP_L_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_R_ELBOW, KP_R_WRIST),
)
# Pares (ombro, cotovelo, pulso) de cada braço, para a análise de postura.
ARM_CHAINS = (
    (KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST),
)


# ---------------------------------------------------------------------------
# Estruturas de dados (DTOs) — resultados de detecção
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """Uma detecção genérica do YOLO já filtrada para nossas classes."""

    class_id: int
    confidence: float
    # Caixa no formato (x1, y1, x2, y2) em pixels (float).
    box: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class PersonDetection(Detection):
    """Pessoa detectada + estado calculado pela regra de negócio."""

    using_phone: bool = False
    # Celular associado (se houver), útil para o Visualizer destacar.
    matched_phone: Detection | None = field(default=None)
    # Keypoints da pose no formato numpy (17, 3) -> [x, y, confiança] por ponto.
    # None quando a pose está desligada ou a pessoa não tem esqueleto estimado.
    keypoints: np.ndarray | None = field(default=None)
    # Pulso (x, y) que foi associado ao celular — usado pelo Visualizer para
    # destacar a mão que está segurando o aparelho. None se não há.
    holding_wrist: tuple[float, float] | None = field(default=None)
    # ID estável de rastreamento (ByteTrack/BoT-SORT) que persiste entre frames.
    # None quando o tracking está desligado, ainda não "esquentou" (cold-start)
    # ou o modelo não fornece IDs (ex.: predict puro / mocks de teste).
    track_id: int | None = field(default=None)
    # Score [0..1] de POSTURA de uso de celular (mão erguida + cotovelo
    # flexionado + cabeça baixa), calculado a partir da pose. Independe de o
    # YOLO ter visto o aparelho — generaliza a detecção. 0.0 sem pose.
    posture_score: float = 0.0
    # True quando o "usando celular" foi decidido SOMENTE pela postura (sem uma
    # caixa de celular associada) — útil para o Visualizer/relatório distinguir.
    by_posture: bool = False


# ---------------------------------------------------------------------------
# Funções geométricas puras (sem estado) — fáceis de testar isoladamente
# ---------------------------------------------------------------------------
def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Área de interseção entre duas caixas (passo 2 do cabeçalho)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Canto superior-esquerdo da interseção = max dos cantos esquerdos.
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    # Canto inferior-direito da interseção = min dos cantos direitos.
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # Largura/altura truncadas em zero (sem sobreposição => negativo).
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Intersection over Union (passo 3 do cabeçalho)."""
    inter = intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def containment_ratio(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """Fração de ``inner`` que está dentro de ``outer`` (passo 4).

    Ex.: ``inner`` = celular, ``outer`` = pessoa. Retorna area_inter /
    area_do_celular. Vale 1.0 se o celular está totalmente contido.
    """
    inter = intersection_area(inner, outer)
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if area_inner <= 0.0:
        return 0.0
    return inter / area_inner


def euclidean_distance(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Distância euclidiana entre os centros das caixas (passo 5)."""
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    # np.hypot = sqrt(dx^2 + dy^2), numericamente estável.
    return float(np.hypot(ax - bx, ay - by))


def point_box_distance(
    px: float, py: float, box: tuple[float, float, float, float]
) -> float:
    """Distância de um ponto (px, py) à caixa ``box``.

    Vale 0.0 quando o ponto está DENTRO da caixa; caso contrário, é a
    distância euclidiana até a borda mais próxima. Usado para medir quão
    perto um pulso está do celular.
    """
    x1, y1, x2, y2 = box
    # Deslocamento em cada eixo: 0 se o ponto já está dentro do intervalo.
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def person_scale(
    keypoints: np.ndarray | None,
    person_box: tuple[float, float, float, float],
    cfg: Config,
) -> float:
    """Referência de escala da pessoa (em pixels), invariante à distância.

    Preferimos a largura entre os ombros (robusta e proporcional ao corpo);
    se um dos ombros não for confiável, caímos para a largura da caixa.
    """
    if keypoints is not None:
        ls = keypoints[KP_L_SHOULDER]
        rs = keypoints[KP_R_SHOULDER]
        if ls[2] >= cfg.wrist_conf_threshold and rs[2] >= cfg.wrist_conf_threshold:
            width = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
            if width > 1.0:
                return width
    return max(1.0, person_box[2] - person_box[0])


def has_confident_wrist(keypoints: np.ndarray | None, cfg: Config) -> bool:
    """True se ao menos um pulso (esq/dir) tem confiança >= limiar."""
    if keypoints is None:
        return False
    return bool(
        keypoints[KP_L_WRIST][2] >= cfg.wrist_conf_threshold
        or keypoints[KP_R_WRIST][2] >= cfg.wrist_conf_threshold
    )


def wrist_phone_proximity(
    keypoints: np.ndarray | None,
    phone_box: tuple[float, float, float, float],
    person_box: tuple[float, float, float, float],
    cfg: Config,
    radius_scale: float = 1.0,
) -> tuple[bool, tuple[float, float] | None, float]:
    """Decide se algum PULSO da pessoa está perto o bastante do celular.

    Retorna ``(segurando, pulso, score)``:
      - ``segurando``: True se o pulso mais próximo está a uma distância
        <= raio (``hand_radius_factor`` × escala da pessoa) da caixa do
        celular.
      - ``pulso``: coordenada (x, y) do pulso vencedor (ou None).
      - ``score``: distância do pulso vencedor (menor = melhor); ``inf`` se
        nenhum pulso confiável está dentro do raio.

    O raio escala com o tamanho da pessoa, então funciona igual para alguém
    perto ou longe da câmera.
    """
    if keypoints is None:
        return (False, None, float("inf"))

    radius = (
        cfg.hand_radius_factor
        * person_scale(keypoints, person_box, cfg)
        * max(1.0, radius_scale)
    )
    best_wrist: tuple[float, float] | None = None
    best_dist = float("inf")
    for idx in (KP_L_WRIST, KP_R_WRIST):
        wx, wy, wconf = keypoints[idx]
        if wconf < cfg.wrist_conf_threshold:
            continue
        dist = point_box_distance(float(wx), float(wy), phone_box)
        if dist <= radius and dist < best_dist:
            best_dist = dist
            best_wrist = (float(wx), float(wy))

    return (best_wrist is not None, best_wrist, best_dist)


def _kp_ok(kp: np.ndarray | None, idx: int, conf_t: float) -> bool:
    """True se o keypoint ``idx`` existe e tem confiança >= ``conf_t``."""
    return kp is not None and idx < len(kp) and float(kp[idx][2]) >= conf_t


def _interior_angle(a, b, c) -> float:
    """Ângulo (graus) no vértice ``b`` do segmento a-b-c. 180° = esticado."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cosang = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def phone_use_posture(
    keypoints: np.ndarray | None,
    person_box: tuple[float, float, float, float],
    cfg: Config,
) -> float:
    """Score [0..1] de "postura de uso de celular" a partir da pose.

    Combina três pistas robustas e independentes da aparência do aparelho
    (logo, não dependem das fotos de calibração):

      1. **Mão erguida à frente do tronco** — o pulso está acima da linha do
         quadril e horizontalmente entre/junto aos ombros (mão trazida ao
         centro do corpo, como quem segura o telefone), não largada ao lado.
      2. **Cotovelo flexionado** — o ângulo ombro-cotovelo-pulso é fechado
         (~< 110°): o antebraço sobe segurando o aparelho.
      3. **Cabeça inclinada para baixo** — o nariz cai abaixo da linha dos
         olhos mais do que no estado ereto (olhando para a tela).

    O score é a média ponderada das pistas disponíveis (a (3) só entra se os
    pontos faciais forem confiáveis). Retorna 0.0 sem pose/ombros confiáveis.
    """
    conf_t = cfg.wrist_conf_threshold
    if not (
        _kp_ok(keypoints, KP_L_SHOULDER, conf_t)
        and _kp_ok(keypoints, KP_R_SHOULDER, conf_t)
    ):
        return 0.0

    ls = keypoints[KP_L_SHOULDER]
    rs = keypoints[KP_R_SHOULDER]
    shoulder_y = (float(ls[1]) + float(rs[1])) / 2.0
    shoulder_w = max(1.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    sx_min, sx_max = sorted((float(ls[0]), float(rs[0])))

    # Linha de referência inferior do tronco: quadris se visíveis, senão a base
    # da caixa (limita a região "acima do quadril").
    hip_ys = [
        float(keypoints[i][1])
        for i in (KP_L_HIP, KP_R_HIP)
        if _kp_ok(keypoints, i, conf_t)
    ]
    hip_y = float(np.mean(hip_ys)) if hip_ys else float(person_box[3])
    torso_h = max(1.0, hip_y - shoulder_y)

    # -- (1)+(2): melhor braço (mão erguida + cotovelo flexionado) ----------
    best_arm = 0.0
    margin = 0.55 * shoulder_w  # tolerância lateral além dos ombros
    for sh, el, wr in ARM_CHAINS:
        if not (_kp_ok(keypoints, wr, conf_t) and _kp_ok(keypoints, sh, conf_t)):
            continue
        wx, wy = float(keypoints[wr][0]), float(keypoints[wr][1])

        # Altura da mão: 0 na linha do quadril, 1 na linha dos ombros (e além).
        raised = (hip_y - wy) / torso_h
        raised_score = float(np.clip(raised / 1.0, 0.0, 1.0))

        # Centralização: mão trazida à frente do corpo (entre os ombros ± margem).
        centered = 1.0 if (sx_min - margin) <= wx <= (sx_max + margin) else 0.0

        # Cotovelo flexionado (precisa do cotovelo para medir o ângulo).
        bend_score = 0.0
        if _kp_ok(keypoints, el, conf_t):
            ang = _interior_angle(
                keypoints[sh][:2], keypoints[el][:2], keypoints[wr][:2]
            )
            # 160°+ (braço esticado) -> 0 ; 70° ou menos (bem dobrado) -> 1.
            bend_score = float(np.clip((160.0 - ang) / 90.0, 0.0, 1.0))

        arm = centered * (0.6 * raised_score + 0.4 * bend_score)
        best_arm = max(best_arm, arm)

    # -- (3): cabeça inclinada para baixo -----------------------------------
    head_score = None
    if _kp_ok(keypoints, KP_NOSE, conf_t) and (
        _kp_ok(keypoints, KP_L_EYE, conf_t) or _kp_ok(keypoints, KP_R_EYE, conf_t)
    ):
        eye_ys = [
            float(keypoints[i][1])
            for i in (KP_L_EYE, KP_R_EYE)
            if _kp_ok(keypoints, i, conf_t)
        ]
        eye_y = float(np.mean(eye_ys))
        # Quanto o nariz está abaixo dos olhos, normalizado pela escala da face
        # (aprox. fração da largura dos ombros). Ereto ~0.05–0.12; olhando para
        # baixo o nariz desce bem mais. Mapeia 0.12→0 .. 0.32→1.
        drop = (float(keypoints[KP_NOSE][1]) - eye_y) / shoulder_w
        head_score = float(np.clip((drop - 0.12) / 0.20, 0.0, 1.0))

    # -- combinação ---------------------------------------------------------
    if head_score is None:
        return float(np.clip(best_arm, 0.0, 1.0))
    # Braço pesa mais (sinal mais discriminante); cabeça reforça.
    return float(np.clip(0.7 * best_arm + 0.3 * head_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Seleção de dispositivo (adaptativa)
# ---------------------------------------------------------------------------
def resolve_device(requested: str = "auto") -> str:
    """Resolve o dispositivo de inferência do torch a usar.

    - Um valor explícito ("cpu", "cuda", "mps", "0", ...) é devolvido como está.
    - "auto" escolhe o melhor disponível na ordem: CUDA (NVIDIA) -> MPS (Apple
      Silicon) -> CPU. O import do torch é LOCAL e protegido: em qualquer falha
      (torch ausente, etc.) caímos para "cpu", nunca quebrando a aplicação.
    """
    if requested and requested != "auto":
        return requested
    try:  # import local: a geometria/os testes não precisam de torch.
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - caminho defensivo
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Detector — wrapper do YOLO + regra de negócio
# ---------------------------------------------------------------------------
class Detector:
    """Encapsula o modelo YOLO e a lógica de "uso de celular".

    O carregamento do modelo é *lazy* (preguiçoso): o peso só é baixado/
    carregado na primeira inferência ou quando ``load()`` é chamado. Isso
    permite instanciar a classe em testes sem precisar do peso real
    (basta injetar um modelo fake via o parâmetro ``model``).
    """

    def __init__(
        self,
        config: Config | None = None,
        model: object | None = None,
        pose_model: object | None = None,
    ) -> None:
        # Injeção de dependência: aceita uma config e (opcionalmente) os
        # modelos já prontos. Em testes, injeta-se mocks; em produção,
        # deixa-se None e os pesos YOLO reais são carregados sob demanda.
        #   - ``model``      : modelo de DETECÇÃO (acha o celular, classe 67).
        #   - ``pose_model`` : modelo de POSE (pessoas + keypoints/braços/mãos).
        self.config = config or settings
        self._model = model
        self._pose_model = pose_model
        # Dispositivo resolvido em load(). Fica None até lá (testes que injetam
        # mocks e não chamam load() seguem com device=None, o que os modelos
        # fake ignoram e o ultralytics real trata como "auto").
        self._device: str | None = None

    # -- ciclo de vida dos modelos -----------------------------------------
    def load(self) -> None:
        """Carrega os pesos YOLO ausentes (detecção e, se ligada, pose)."""
        # Resolve o dispositivo uma única vez (cuda -> mps -> cpu, ou explícito).
        self._device = resolve_device(self.config.device)
        if self._model is None:
            # Import local para não exigir ultralytics em quem só usa a
            # geometria (e para não pesar a importação do pacote/testes).
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_path)
        if self.config.pose_enabled and self._pose_model is None:
            from ultralytics import YOLO

            self._pose_model = YOLO(self.config.pose_model_path)

    @property
    def model(self) -> object:
        """Modelo de detecção (celular), garantindo que esteja carregado."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def pose_model(self) -> object:
        """Modelo de pose (pessoas + keypoints), garantindo carregamento."""
        if self._pose_model is None:
            self.load()
        return self._pose_model

    # -- parsing dos resultados do ultralytics -----------------------------
    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame deve ser um numpy.ndarray válido.")

    @staticmethod
    def _parse_results(results, cfg: Config) -> list[Detection]:
        """Converte o objeto Results do ultralytics em ``Detection``.

        Aceita tanto o formato real do ultralytics (lista de Results com
        ``.boxes`` contendo tensores) quanto estruturas já simplificadas,
        sendo tolerante para facilitar os testes.
        """
        detections: list[Detection] = []
        if not results:
            return detections

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            # boxes.xyxy, boxes.cls, boxes.conf são tensores (torch).
            # Convertemos para numpy de forma defensiva.
            xyxy = _to_numpy(getattr(boxes, "xyxy", None))
            cls = _to_numpy(getattr(boxes, "cls", None))
            conf = _to_numpy(getattr(boxes, "conf", None))
            if xyxy is None or cls is None or conf is None:
                continue

            for i in range(len(xyxy)):
                class_id = int(cls[i])
                if class_id not in (cfg.person_class_id, cfg.phone_class_id):
                    continue
                x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
                detections.append(
                    Detection(
                        class_id=class_id,
                        confidence=float(conf[i]),
                        box=(x1, y1, x2, y2),
                    )
                )
        return detections

    @staticmethod
    def _parse_pose_results(results, cfg: Config) -> list[PersonDetection]:
        """Converte os Results do modelo de POSE em ``PersonDetection``.

        Lê as caixas das pessoas (``result.boxes``) e os keypoints
        (``result.keypoints``), juntando cada pessoa ao seu esqueleto.
        """
        people: list[PersonDetection] = []
        if not results:
            return people

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_numpy(getattr(boxes, "xyxy", None))
            cls = _to_numpy(getattr(boxes, "cls", None))
            conf = _to_numpy(getattr(boxes, "conf", None))
            if xyxy is None or cls is None or conf is None:
                continue

            # IDs de rastreamento: presentes só quando veio de model.track(...).
            # Em predict puro (ou mocks) o atributo não existe / é None -> None.
            ids = _to_numpy(getattr(boxes, "id", None))

            kp_all = _extract_keypoints(getattr(result, "keypoints", None))

            for i in range(len(xyxy)):
                if int(cls[i]) != cfg.person_class_id:
                    continue
                x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
                kp = (
                    kp_all[i]
                    if kp_all is not None and i < len(kp_all)
                    else None
                )
                track_id = (
                    int(ids[i]) if ids is not None and i < len(ids) else None
                )
                people.append(
                    PersonDetection(
                        class_id=cfg.person_class_id,
                        confidence=float(conf[i]),
                        box=(x1, y1, x2, y2),
                        keypoints=kp,
                        track_id=track_id,
                    )
                )
        return people

    # -- inferência (entrypoints sobreponíveis) ----------------------------
    def detect_phones(self, frame: np.ndarray) -> list[Detection]:
        """Detecta apenas os celulares no frame (modelo de detecção)."""
        self._validate_frame(frame)
        cfg = self.config
        results = self.model.predict(
            frame,
            conf=cfg.phone_confidence_threshold,
            iou=cfg.nms_iou_threshold,
            classes=[cfg.phone_class_id],
            imgsz=cfg.imgsz,
            device=self._device,
            verbose=False,
        )
        return [
            d
            for d in self._parse_results(results, cfg)
            if d.class_id == cfg.phone_class_id
        ]

    def detect_people(self, frame: np.ndarray) -> list[PersonDetection]:
        """Detecta as pessoas + seus keypoints (modelo de pose)."""
        self._validate_frame(frame)
        cfg = self.config
        results = self.pose_model.predict(
            frame,
            conf=cfg.confidence_threshold,
            iou=cfg.nms_iou_threshold,
            classes=[cfg.person_class_id],
            imgsz=cfg.imgsz,
            device=self._device,
            verbose=False,
        )
        return self._parse_pose_results(results, cfg)

    def detect_people_tracked(self, frame: np.ndarray) -> list[PersonDetection]:
        """Como ``detect_people``, mas com rastreamento entre frames.

        Usa ``pose_model.track(persist=True, tracker=...)`` (ByteTrack/BoT-SORT
        embutidos no ultralytics) para atribuir um ``track_id`` estável a cada
        pessoa. O ``persist=True`` mantém o estado do tracker entre chamadas
        consecutivas — por isso esta instância de Detector deve processar os
        frames em ordem. No primeiro frame (cold-start) os IDs podem vir nulos.
        """
        self._validate_frame(frame)
        cfg = self.config
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker=cfg.tracker_config,
            conf=cfg.confidence_threshold,
            iou=cfg.nms_iou_threshold,
            classes=[cfg.person_class_id],
            imgsz=cfg.imgsz,
            device=self._device,
            verbose=False,
        )
        return self._parse_pose_results(results, cfg)

    # -- regra de negócio ---------------------------------------------------
    def associate(
        self,
        people: list[PersonDetection],
        phones: list[Detection],
    ) -> list[PersonDetection]:
        """Decide quais pessoas estão SEGURANDO/USANDO um celular.

        Para cada celular:
          1. (Sinal primário) Pulso-proximidade: entre as pessoas cujo PULSO
             cai dentro do raio do celular, vence a de pulso mais próximo. O
             raio é AMPLIADO quando a POSTURA da pessoa indica uso (mão erguida
             + cotovelo dobrado + cabeça baixa), o que recupera celulares
             escuros/borrados que saem com a caixa ligeiramente fora do raio.
          2. (Fallback) Contenção: se NENHUM pulso confiável está perto, só aí
             usamos a geometria antiga (IoU/contenção sobre a caixa inteira),
             e apenas para pessoas cujos pulsos não são visíveis — assim, ver o
             pulso longe do celular significa "não está segurando" (precisão).
          3. (Sinal de POSTURA autônomo) Mesmo sem caixa de celular, marca uso
             quando a postura é fortemente típica (``posture_standalone_threshold``)
             — generaliza além do que o detector COCO consegue ver.
        """
        cfg = self.config

        # Postura calculada UMA vez por pessoa (independe do celular). Vira
        # tanto reforço do raio (passo 1) quanto sinal autônomo (passo 3).
        for person in people:
            person.posture_score = (
                phone_use_posture(person.keypoints, person.box, cfg)
                if cfg.posture_enabled
                else 0.0
            )

        for phone in phones:
            # -- Passo 1: pulso-proximidade (raio reforçado pela postura) ---
            best_person: PersonDetection | None = None
            best_wrist: tuple[float, float] | None = None
            best_score = float("inf")
            for person in people:
                # Postura forte amplia o raio de tolerância pulso↔celular.
                scale = 1.0
                if person.posture_score >= cfg.posture_assist_threshold:
                    scale = 1.0 + cfg.posture_radius_bonus * person.posture_score
                holding, wrist, score = wrist_phone_proximity(
                    person.keypoints, phone.box, person.box, cfg, radius_scale=scale
                )
                if holding and score < best_score:
                    best_score = score
                    best_person = person
                    best_wrist = wrist

            # -- Passo 2: fallback de contenção (sem pulso visível) --------
            if best_person is None:
                best_distance = float("inf")
                for person in people:
                    # Se vemos o pulso da pessoa e ele NÃO ficou perto (passo
                    # 1), confiamos nisso: ela não está segurando. Não cair no
                    # fallback é justamente o ganho de precisão.
                    if has_confident_wrist(person.keypoints, cfg):
                        continue
                    overlap_iou = iou(person.box, phone.box)
                    contained = containment_ratio(
                        inner=phone.box, outer=person.box
                    )
                    if (
                        overlap_iou >= cfg.association_iou_threshold
                        or contained >= cfg.containment_threshold
                    ):
                        dist = euclidean_distance(person.box, phone.box)
                        if dist < best_distance:
                            best_distance = dist
                            best_person = person
                            best_wrist = None

            if best_person is not None:
                best_person.using_phone = True
                best_person.by_posture = False
                best_person.holding_wrist = best_wrist
                # Mantém apenas o celular mais "forte" se já houver um.
                if best_person.matched_phone is None or (
                    phone.confidence > best_person.matched_phone.confidence
                ):
                    best_person.matched_phone = phone

        # -- Passo 3: postura autônoma (sem caixa de celular) ---------------
        # Para quem ainda não foi marcado por um celular detectado, a postura
        # muito típica de uso já basta. Limiar alto + suavização temporal
        # contêm falsos positivos.
        if cfg.posture_enabled:
            for person in people:
                if not person.using_phone and (
                    person.posture_score >= cfg.posture_standalone_threshold
                ):
                    person.using_phone = True
                    person.by_posture = True

        return people

    def process_frame(self, frame: np.ndarray) -> list[PersonDetection]:
        """Pipeline completo: detectar + associar. Retorna pessoas com estado.

        Com pose LIGADA: pessoas (pose) + celulares (detecção) → associação
        por pulso. Com pose DESLIGADA: um único modelo detecta pessoa+celular
        e a associação cai no fallback de contenção (comportamento antigo,
        mais leve).
        """
        self._validate_frame(frame)
        cfg = self.config
        if not cfg.pose_enabled:
            # Roda no limiar mais baixo (do celular) para não perder o aparelho
            # e, depois, exige o limiar maior só para a PESSOA — assim mantemos
            # recall do celular sem afrouxar a detecção de pessoas.
            results = self.model.predict(
                frame,
                conf=cfg.phone_confidence_threshold,
                iou=cfg.nms_iou_threshold,
                classes=[cfg.person_class_id, cfg.phone_class_id],
                imgsz=cfg.imgsz,
                device=self._device,
                verbose=False,
            )
            detections = self._parse_results(results, cfg)
            people = [
                PersonDetection(d.class_id, d.confidence, d.box)
                for d in detections
                if d.class_id == cfg.person_class_id
                and d.confidence >= cfg.confidence_threshold
            ]
            phones = [
                d for d in detections if d.class_id == cfg.phone_class_id
            ]
            return self.associate(people, phones)

        people = self.detect_people(frame)
        phones = self.detect_phones(frame)
        return self.associate(people, phones)

    def process_frame_tracked(self, frame: np.ndarray) -> list[PersonDetection]:
        """Igual a ``process_frame`` (caminho com pose), mas com rastreamento.

        Usa ``detect_people_tracked`` para obter pessoas com ``track_id``
        estável. Requer pose LIGADA — o orquestrador (``main``) só ativa este
        caminho quando ``pose_enabled`` e ``tracking_enabled`` são verdadeiros.
        """
        self._validate_frame(frame)
        people = self.detect_people_tracked(frame)
        phones = self.detect_phones(frame)
        return self.associate(people, phones)


# ---------------------------------------------------------------------------
# Utilitário interno
# ---------------------------------------------------------------------------
def _to_numpy(value):
    """Converte tensores torch / listas em numpy de forma tolerante."""
    if value is None:
        return None
    # Tensores torch possuem .cpu().numpy(); numpy já é numpy; listas viram
    # array. Tratamos tudo de forma defensiva para não acoplar a torch.
    if hasattr(value, "cpu"):
        try:
            value = value.cpu().numpy()
        except Exception:  # pragma: no cover - caminho defensivo
            value = np.asarray(value)
    return np.asarray(value)


def _extract_keypoints(kp_obj) -> np.ndarray | None:
    """Extrai os keypoints da pose como numpy (N, 17, 3) = [x, y, conf].

    Tolerante ao formato do ultralytics (``keypoints.data`` (N,17,3) ou
    ``keypoints.xy`` (N,17,2) + ``keypoints.conf`` (N,17)) e a mocks de teste.
    Retorna None quando não há keypoints utilizáveis.
    """
    if kp_obj is None:
        return None

    # Caminho preferido: .data já traz [x, y, conf] por ponto.
    data = _to_numpy(getattr(kp_obj, "data", None))
    if data is not None and data.ndim == 3 and data.shape[2] >= 3:
        return data[:, :, :3].astype(float)

    # Alternativa: .xy (coordenadas) + .conf (confiança) separados.
    xy = _to_numpy(getattr(kp_obj, "xy", None))
    if xy is not None and xy.ndim == 3 and xy.shape[2] >= 2:
        n, k = xy.shape[0], xy.shape[1]
        out = np.zeros((n, k, 3), dtype=float)
        out[:, :, :2] = xy[:, :, :2]
        conf = _to_numpy(getattr(kp_obj, "conf", None))
        out[:, :, 2] = conf if conf is not None else 1.0
        return out

    return None
