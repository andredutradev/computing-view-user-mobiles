# Computing View — Detecção de Uso de Celular

Sistema de visão computacional que analisa um feed de câmera em tempo real
(ou um arquivo de vídeo) e detecta se as pessoas na cena estão
**segurando/usando aparelhos celulares**.

A detecção combina **dois modelos YOLOv8** *nano* pré-treinados no **COCO**:

- **Pose** (`yolov8n-pose.pt`) — localiza as pessoas **e** os keypoints do
  esqueleto: ombros, cotovelos e **pulsos** (ou seja, **braços e mãos**).
- **Detecção** (`yolov8n.pt`) — localiza os **celulares** (classe `cell phone`).

A inferência de "está segurando o celular" é feita pela **proximidade do
pulso (mão) ao celular** — não mais pela caixa inteira da pessoa. Isso elimina
falsos positivos (celular perto do pé/cintura, ou de uma pessoa ao lado). Há um
**fallback** por contenção (IoU + % do celular dentro da pessoa) para quando os
pulsos não estão visíveis. Pessoas **segurando celular ficam com bounding box
verde** (com um "alvo": cantos em L + crosshair); as demais, **laranja**. Os
**braços** são desenhados em ciano e as **mãos** (pulsos) em magenta — a mão que
segura o aparelho fica destacada em vermelho.

> Modo leve: `--no-pose` (ou `CVUM_POSE=0`) desliga a pose e volta ao
> comportamento antigo de modelo único + contenção (mais rápido em máquinas
> fracas).

## Modo demo (abre a interface sem YOLO/câmera)

Para ver a interface funcionando na hora — sem baixar pesos nem ter webcam —
use o modo demo, que gera uma cena sintética (uma pessoa e um celular que se
aproxima e a sobrepõe) e roda a **mesma** lógica de associação de produção:

```bash
python3 -m src.main --demo
```

O box da pessoa alterna para **verde** quando o celular a sobrepõe. Para
rodar sem abrir janela (CI/validação): `python3 -m src.main --demo --no-display --max-frames 160`.

## Stack

- Python 3.10+
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`) — detecção **e** pose (esqueleto)
- NumPy
- Pytest

## Arquitetura

O projeto segue princípios **SOLID** e separa responsabilidades em camadas
desacopladas:

| Módulo               | Responsabilidade                                            |
|----------------------|-------------------------------------------------------------|
| `src/config.py`      | Configuração global imutável (limiares, classes, cores).    |
| `src/video_source.py`| **Strategy + Factory** para a fonte de vídeo (webcam/arquivo). |
| `src/detector.py`    | Wrappers do YOLO (pose + detecção) + regra de negócio (associação pulso↔celular). |
| `src/visualizer.py`  | Desenho de bounding boxes, esqueleto (braços/mãos), rótulos e HUD. |
| `src/main.py`        | Loop principal — orquestra as camadas via injeção de dependência. |

### Padrão Strategy / Factory para a fonte de vídeo

`VideoSource` é a interface abstrata. `WebcamSource` e `FileSource` são as
estratégias concretas, e `VideoSourceFactory` escolhe qual instanciar a
partir da configuração. O loop de detecção depende apenas da interface —
trocar webcam por arquivo **não** exige alterar a lógica principal.

## Lógica matemática da sobreposição

Cada bounding box é `(x1, y1, x2, y2)` em pixels. Para decidir se uma pessoa
está usando um celular combinamos três métricas (detalhadas em
`src/detector.py`):

1. **Área da interseção** — retângulo de sobreposição:
   `iw = max(0, min(Ax2,Bx2) - max(Ax1,Bx1))`,
   `ih = max(0, min(Ay2,By2) - max(Ay1,By1))`, `area = iw * ih`.
2. **IoU** = `area_inter / (area_A + area_B - area_inter)` — clássico, mas
   baixo quando os objetos têm tamanhos muito diferentes.
3. **Contenção** = `area_inter / area_celular` — fração do celular dentro da
   pessoa; robusta para o nosso caso (celular << pessoa).
4. **Distância euclidiana** entre centros — desempata quando há várias
   pessoas candidatas, atribuindo o celular à mais próxima.

**Decisão (sinal primário — pulso/mão):** com a pose ligada, a pessoa é marcada
como *segurando celular* quando algum **pulso** (esquerdo ou direito, com
confiança ≥ `wrist_conf_threshold`) está a uma distância ≤ **raio** da caixa do
celular. O raio = `hand_radius_factor` × escala da pessoa (largura dos ombros;
na falta, largura da caixa), o que torna a regra **invariante à distância** da
câmera. Entre várias pessoas/pulsos candidatos, vence o **pulso mais próximo**.

**Decisão (fallback — contenção):** se nenhum pulso confiável está perto, e
apenas para pessoas cujos pulsos **não** são visíveis, recai-se na regra antiga:
`IoU ≥ limiar` **ou** `contenção ≥ limiar`. Importante: se o pulso É visível e
está longe, confiamos nisso (a pessoa **não** está segurando) — é justamente aí
que mora o ganho de precisão. Todos os limiares ficam em `config.py`.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```

Na primeira execução os pesos do YOLO são baixados automaticamente:
`yolov8n.pt` (detecção do celular) e `yolov8n-pose.pt` (pose — braços e mãos).
Nenhuma dependência nova é necessária: a pose já vem no `ultralytics`.

## Execução

```bash
# Demo (abre a janela na hora, sem YOLO/câmera) — recomendado para testar
python3 -m src.main --demo

# Webcam (tempo real) — padrão
python3 -m src.main

# Arquivo de vídeo
python3 -m src.main --source file --video data/sample_video.mp4

# Gravar a saída anotada em um arquivo (ex.: para rever depois)
python3 -m src.main --demo --no-display --max-frames 160 --save data/demo_output.mp4

# Ajustes
python3 -m src.main --model yolov8s.pt --conf 0.5

# Modo leve: desliga a pose (braços/mãos) e usa só contenção
python3 -m src.main --no-pose
```

Pressione **`q`** ou **ESC** para sair.

Também é possível configurar por variáveis de ambiente:
`CVUM_SOURCE`, `CVUM_VIDEO_PATH`, `CVUM_MODEL_PATH`, `CVUM_CONF`,
`CVUM_CONTAINMENT`, `CVUM_WEBCAM_INDEX` e, para a pose: `CVUM_POSE` (0/1),
`CVUM_POSE_MODEL`, `CVUM_WRIST_CONF` (confiança mínima do pulso) e
`CVUM_HAND_RADIUS` (raio de proximidade pulso↔celular, fração da escala).

## Testes

Os testes mockam o YOLO e os frames do OpenCV — não baixam pesos nem exigem
câmera:

```bash
python3 -m pytest -v
```

## Estrutura

```
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   ├── demo.py
│   ├── video_source.py
│   ├── visualizer.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_demo.py
├── data/
│   └── sample_video.mp4      # (adicionar manualmente)
├── requirements.txt
└── README.md
```
