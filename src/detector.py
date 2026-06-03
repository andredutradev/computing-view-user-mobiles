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
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
# Segmentos (ombro→cotovelo→pulso) de cada lado, para desenhar os braços.
ARM_SEGMENTS = (
    (KP_L_SHOULDER, KP_L_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_R_ELBOW, KP_R_WRIST),
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

    radius = cfg.hand_radius_factor * person_scale(keypoints, person_box, cfg)
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

    # -- ciclo de vida dos modelos -----------------------------------------
    def load(self) -> None:
        """Carrega os pesos YOLO ausentes (detecção e, se ligada, pose)."""
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
                people.append(
                    PersonDetection(
                        class_id=cfg.person_class_id,
                        confidence=float(conf[i]),
                        box=(x1, y1, x2, y2),
                        keypoints=kp,
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
            conf=cfg.confidence_threshold,
            iou=cfg.nms_iou_threshold,
            classes=[cfg.phone_class_id],
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
            verbose=False,
        )
        return self._parse_pose_results(results, cfg)

    # -- regra de negócio ---------------------------------------------------
    def associate(
        self,
        people: list[PersonDetection],
        phones: list[Detection],
    ) -> list[PersonDetection]:
        """Decide quais pessoas estão SEGURANDO um celular.

        Para cada celular:
          1. (Sinal primário) Pulso-proximidade: entre as pessoas cujo PULSO
             cai dentro do raio do celular, vence a de pulso mais próximo.
          2. (Fallback) Contenção: se NENHUM pulso confiável está perto, só aí
             usamos a geometria antiga (IoU/contenção sobre a caixa inteira),
             e apenas para pessoas cujos pulsos não são visíveis — assim, ver o
             pulso longe do celular significa "não está segurando" (precisão).
        """
        cfg = self.config

        for phone in phones:
            # -- Passo 1: pulso-proximidade --------------------------------
            best_person: PersonDetection | None = None
            best_wrist: tuple[float, float] | None = None
            best_score = float("inf")
            for person in people:
                holding, wrist, score = wrist_phone_proximity(
                    person.keypoints, phone.box, person.box, cfg
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
                best_person.holding_wrist = best_wrist
                # Mantém apenas o celular mais "forte" se já houver um.
                if best_person.matched_phone is None or (
                    phone.confidence > best_person.matched_phone.confidence
                ):
                    best_person.matched_phone = phone

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
            results = self.model.predict(
                frame,
                conf=cfg.confidence_threshold,
                iou=cfg.nms_iou_threshold,
                classes=[cfg.person_class_id, cfg.phone_class_id],
                verbose=False,
            )
            detections = self._parse_results(results, cfg)
            people = [
                PersonDetection(d.class_id, d.confidence, d.box)
                for d in detections
                if d.class_id == cfg.person_class_id
            ]
            phones = [
                d for d in detections if d.class_id == cfg.phone_class_id
            ]
            return self.associate(people, phones)

        people = self.detect_people(frame)
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
