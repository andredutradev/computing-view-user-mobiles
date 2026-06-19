# Computing View — Detecção de Uso de Celular

Sistema de visão computacional que analisa um feed de câmera em tempo real
(ou um arquivo de vídeo) e detecta se as pessoas na cena estão
**segurando/usando aparelhos celulares**.

A detecção combina **dois modelos YOLO11** pré-treinados no **COCO**:

- **Pose** (`yolo11s-pose.pt`) — localiza as pessoas **e** os keypoints do
  esqueleto: ombros, cotovelos e **pulsos** (ou seja, **braços e mãos**).
- **Detecção** (`yolo11m.pt`) — localiza os **celulares** (classe `cell phone`).
  O modelo *medium* é o padrão porque celulares **escuros/borrados na mão** —
  o caso real mais difícil — passam despercebidos pelos modelos *nano/small*.
  A inferência roda em `--imgsz 960` e o celular tem um **limiar de confiança
  próprio, mais baixo** (`CVUM_PHONE_CONF`, padrão 0.25), já que sai com menos
  confiança que a pessoa. Em máquinas fracas, volte ao leve com
  `CVUM_MODEL_PATH=yolov8n.pt CVUM_POSE_MODEL=yolov8n-pose.pt CVUM_IMGSZ=640`.

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

## Consistência do estado (sem piscar com a movimentação)

Para o estado **"usando celular" não oscilar** (laranja↔verde) a cada frame
quando a pessoa se mexe, o sistema rastreia cada pessoa entre frames
(**ByteTrack/BoT-SORT**, embutidos no `ultralytics`) e aplica **suavização
temporal com histerese** por `track_id`:

- voto em **janela deslizante** das decisões cruas (`smoothing_window` frames);
- **histerese**: liga em `on_threshold` (0.6) e só desliga em `off_threshold`
  (0.35) — a "zona morta" entre os dois elimina o tremido;
- **grace period**: o box sobrevive a oclusões/sumiços breves
  (`grace_period` frames) em vez de desaparecer.

Tudo é configurável (ver `config.py` / variáveis `CVUM_*`) e pode ser desligado
com `--no-track` / `--no-smooth`. O **dispositivo** é escolhido automaticamente
(`--device auto` → CUDA → MPS/Apple → CPU); para mais precisão em objetos
pequenos (o celular), aumente `--imgsz` (ex.: 960/1280) ou troque o modelo
(`--model yolo11s.pt`).

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

- Python 3.9+ (o código usa `from __future__ import annotations`)
- OpenCV (`opencv-python`)
- Pillow — renderização de texto com **acentuação** correta sobre o vídeo
- Ultralytics YOLO (`ultralytics`) — detecção **e** pose (esqueleto)
- PySide6 (Qt) — painel de controle desktop (interface principal)
- NumPy
- Pytest

> **Acentuação no vídeo.** O `cv2.putText` usa fontes Hershey (ASCII) e troca
> acentos por "?". Todo o texto desenhado sobre os frames (rótulos, HUD, painel
> OpenCV) passa por `src/text_render.py`, que desenha com Pillow/TrueType
> (fonte empacotada em `assets/fonts/DejaVuSans.ttf`) — "André", "Presença",
> "Usando Celular" aparecem corretos.

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

### Sinal de POSTURA — generaliza além de "ver o aparelho"

Depender só de o YOLO enxergar o celular falha quando o aparelho está **escuro,
borrado ou oculto na mão** — o caso real mais comum. Por isso há um sinal de
**postura** (`src/detector.py:phone_use_posture`), estimado da própria pose e
**independente da aparência do celular** (logo, não depende das fotos de
calibração). Combina três pistas:

1. **mão erguida à frente do tronco** (pulso acima da linha do quadril e
   horizontalmente entre/junto aos ombros);
2. **cotovelo flexionado** (ângulo ombro-cotovelo-pulso fechado);
3. **cabeça inclinada para baixo** (nariz abaixo da linha dos olhos — olhando
   para a tela).

O score `[0..1]` é usado de duas formas: (a) **reforça** o raio pulso↔celular
quando a postura é típica (recupera celulares fracos — `posture_assist_threshold`);
(b) marca uso **só pela postura**, sem caixa de celular, quando o score é alto
(`posture_standalone_threshold`). A suavização temporal/histerese contém falsos
positivos. No overlay, esse caso aparece como **"Usando Celular (postura)"**.
Desligue com `CVUM_POSTURE=0`.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```

Na primeira execução os pesos do YOLO são baixados automaticamente:
`yolo11m.pt` (detecção do celular) e `yolo11s-pose.pt` (pose — braços e mãos).
Nenhuma dependência nova é necessária: a pose já vem no `ultralytics`.

## Docker

O projeto pode rodar com um único `Dockerfile`. Não há necessidade de
`docker compose` para a execução padrão, porque a aplicação não depende de
banco, fila, proxy ou outro serviço externo.

```bash
docker build -t computing-view-user-mobiles .
```

A imagem instala PyTorch em modo CPU por padrão. Se você precisar de uma base
com GPU/CUDA, ajuste o `Dockerfile`/imagem base conforme o runtime NVIDIA do
ambiente.

Validação headless, sem câmera e sem YOLO:

```bash
docker run --rm computing-view-user-mobiles
```

Rodar com vídeo local e gravar saída anotada:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  computing-view-user-mobiles \
  --source file --video data/sample_video.mp4 --no-display --save data/output.mp4
```

Para webcam no Linux, exponha o dispositivo:

```bash
docker run --rm --device /dev/video0:/dev/video0 \
  -v "$PWD/data:/app/data" \
  computing-view-user-mobiles --source webcam
```

Os pesos `.pt`, vídeos e dados em `data/` ficam fora do contexto de build por
padrão via `.dockerignore`, evitando imagens grandes e inclusão acidental de
dados biométricos. Monte `data/` como volume quando precisar persistir vídeos,
galerias de alunos, relatórios ou modelos baixados em tempo de execução.

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

### Rastreamento, suavização e dispositivo

```bash
# Trocar o tracker, desligar a suavização, forçar CPU, subir a resolução
python3 -m src.main --source file --video data/sample_video.mp4 \
    --tracker botsort --device cpu --imgsz 960
python3 -m src.main --no-track     # sem track_id estável
python3 -m src.main --no-smooth    # sem histerese (volta a piscar)
```

## Sistema de presença em sala (reconhecimento facial)

Além de detectar o uso de celular, o sistema **identifica cada aluno** por
reconhecimento facial (**InsightFace**), acompanha **presença e movimentação**
em tempo real, **contabiliza o tempo de permanência** (dwell time) e gera
**relatórios de frequência e ocupação** (CSV + PDF) baixáveis.

Robustez do reconhecimento **a partir de uma única foto** (ajustes em
`config.py`): a matrícula faz **augmentation** (embute o espelho horizontal,
extraindo mais de um embedding por foto — `face_enroll_augment`) e um **gate de
qualidade** (rejeita rostos com `det_score` baixo — `face_enroll_min_det_score`);
o reconhecimento usa **Test-Time Augmentation** (média do rosto e do seu espelho
— `face_tta_flip`), agrega por **aluno** (não por foto) e aplica uma **margem
anti-ambiguidade** entre o 1º e o 2º melhor aluno (`face_match_margin`).

Dependências extras (já no `requirements.txt`): `insightface`, `onnxruntime`
(para GPU NVIDIA, use `onnxruntime-gpu` — **nunca** os dois juntos) e `fpdf2`.
O pacote de modelos `buffalo_l` (~300 MB) é baixado no primeiro uso.

**1) Matrícula dos alunos** — uma foto de referência por aluno em
`data/students/` (subpasta por aluno permite várias fotos, o que melhora a
robustez):

```
data/students/
├── ana_silva/        # subpasta = nome do aluno
│   ├── frente.jpg
│   └── lado.jpg
└── bruno_costa.jpg   # ou imagem "plana" (o nome do arquivo vira o id)
```

```bash
python3 -m src.main --enroll data/students      # gera data/students/gallery.npz
# (equivalente: python3 -m src.attendance.enrollment --photos data/students)
```

**2) Monitorar (vídeo ou webcam) com presença ligada:**

```bash
# Sobe um vídeo e abre a janela como se fosse a câmera ao vivo, com presença
python3 -m src.main --source file --video data/sample_video.mp4 --attendance

# Relatórios são gravados ao final em data/reports/<fonte>_<timestamp>/
python3 -m src.main --attendance --report-out data/reports
```

**3) Painel de controle desktop (PySide6/Qt)** — faz tudo por uma interface
moderna, com acentos nativos e o **vídeo embutido na janela**:

```bash
python3 -m src.main --ui
```

A janela tem três blocos:

1. **Matrícula** — campo de **nome** + **Ligar webcam** (preview ao vivo) +
   **Capturar e matricular**: tira a foto da câmera, detecta o rosto, salva a
   foto de referência em `data/students/<slug>/` e matricula na hora
   (acumulando amostras se o aluno já existir). Há também **Matricular pasta
   data/students** para a matrícula em lote a partir de fotos já existentes.
2. **Monitoramento** — escolha webcam/arquivo, ajuste a **sensibilidade do
   reconhecimento** e clique **Iniciar**/**Parar**. O vídeo anotado (boxes,
   esqueleto, identidades) aparece dentro da própria janela.
3. **Relatório** — **Gerar relatório** (CSV + PDF) e abre a pasta de saída.

> **Backends e fallback.** O `--ui` usa o **Qt (PySide6)** quando disponível
> (recomendado). Sem PySide6, cai para Tkinter (se o Tk renderizar bem, ≥ 8.6)
> e, em último caso (ex.: Tk 8.5 do macOS, que abre a janela preta), para um
> **painel OpenCV** equivalente (botões clicáveis + atalhos `I`/`R`/`M`/`Q`).
> Force com `CVUM_UI_BACKEND=qt` | `tk` | `cv`.

Os relatórios gerados:

| Arquivo            | Conteúdo                                                        |
|--------------------|-----------------------------------------------------------------|
| `frequencia.csv/pdf` | Por pessoa: presente?, tempo total, **tempo usando o celular** (`phone_hms`) e **% do tempo no celular** (`pct_phone`), 1º/último avistamento, % da sessão. Alunos matriculados nunca vistos aparecem como **ausentes**; quem usa o celular sem estar matriculado aparece como `Pessoa #ID`. |
| `ocupacao.csv/pdf`   | Série temporal (pessoas, identificados, usando celular) + pico e média de ocupação. |

> **Tempo de celular por pessoa.** Para creditar o tempo a um nome (e não a
> `Pessoa #ID`), basta matricular o aluno (passo 1). Tracks não identificados
> que usam o celular ainda entram no relatório como anônimos por ID.

## Calibrar a precisão com imagens (sem abrir a câmera)

Para medir e ajustar a detecção de celular contra **exemplos reais**, coloque
fotos de pessoas usando o celular numa pasta e rode o validador — ele processa
cada imagem com a mesma regra de produção e reporta quantas foram reconhecidas
como uso de celular:

```bash
python3 -m src.main --eval-images training_true_examples
```

```text
imagem                                 pessoas  cel   conf postura  usando?
Foto ...15.34.jpg                            3    1   0.59    0.25      SIM
Foto ...15.36.jpg                            2    1   0.32    0.41      SIM
Acertos (usando celular): 3/3  (100%)
```

A coluna **postura** mostra o score do sinal de pose; um acerto marcado `SIM*`
foi confirmado **só pela postura**, sem caixa de celular detectada. Se alguma
imagem sair como `NAO`, baixe o limiar do celular (`CVUM_PHONE_CONF`), suba a
resolução (`CVUM_IMGSZ`), afrouxe a regra da mão
(`CVUM_HAND_RADIUS`/`CVUM_WRIST_CONF`) ou a postura
(`CVUM_POSTURE_SOLO`/`CVUM_POSTURE_ASSIST`) e rode de novo.

## Privacidade (LGPD)

Os *embeddings* faciais dos alunos são **dado pessoal sensível** (LGPD,
Lei 13.709/2018, art. 5º II e art. 11). Antes de usar em produção:

- obtenha **consentimento explícito** de cada aluno (ou do responsável, se
  menor) e limite o uso à finalidade de controle de presença;
- a pasta `data/students/` e a galeria (`gallery.npz`/`.json`) contêm dados
  biométricos — mantenha-as **fora de repositórios/backups compartilhados**
  (já estão no `.gitignore`), com acesso restrito e, idealmente, cifradas;
- defina uma **política de retenção/exclusão**; `Gallery.remove(student_id)`
  atende ao direito de eliminação.

Isto é um alerta para a equipe, **não** aconselhamento jurídico.

Também é possível configurar por variáveis de ambiente:
`CVUM_SOURCE`, `CVUM_VIDEO_PATH`, `CVUM_MODEL_PATH`, `CVUM_IMGSZ`, `CVUM_CONF`,
`CVUM_PHONE_CONF` (confiança mínima só do celular), `CVUM_CONTAINMENT`,
`CVUM_WEBCAM_INDEX` e, para a pose: `CVUM_POSE` (0/1), `CVUM_POSE_MODEL`,
`CVUM_WRIST_CONF` (confiança mínima do pulso) e `CVUM_HAND_RADIUS` (raio de
proximidade pulso↔celular, fração da escala).

Postura: `CVUM_POSTURE` (0/1), `CVUM_POSTURE_ASSIST`, `CVUM_POSTURE_SOLO`,
`CVUM_POSTURE_RADIUS_BONUS`. Reconhecimento facial: `CVUM_FACE_THRESH`,
`CVUM_FACE_MARGIN`, `CVUM_FACE_TTA`, `CVUM_FACE_ENROLL_AUG`,
`CVUM_FACE_ENROLL_MIN_DET`. Interface: `CVUM_UI_BACKEND` (`qt`/`tk`/`cv`) e
`CVUM_FONT_PATH` (fonte TrueType alternativa para os acentos no vídeo).

## Testes

Os testes mockam o YOLO e os frames do OpenCV — não baixam pesos nem exigem
câmera:

```bash
python3 -m pytest -v
```

## Estrutura

```text
├── assets/
│   └── fonts/               # DejaVuSans.ttf (acentos no vídeo, empacotada)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py          # YOLO (pose+detecção) + tracking + postura + regra
│   ├── temporal.py          # suavização temporal/histerese (consistência)
│   ├── text_render.py       # texto Unicode (acentos) sobre os frames (Pillow)
│   ├── eval_images.py       # validador de detecção em imagens estáticas
│   ├── demo.py
│   ├── video_source.py
│   ├── visualizer.py
│   ├── main.py
│   └── attendance/          # sistema de presença em sala
│       ├── enrollment.py    # matrícula (galeria, augmentation, captura única)
│       ├── face_recognizer.py  # InsightFace + TTA + cache de identidade
│       ├── geometry.py      # recorte da cabeça + similaridade de cosseno
│       ├── attendance.py    # presença/movimentação em tempo real
│       ├── session.py       # modelo de dados (dwell time, ocupação)
│       ├── reports.py       # relatórios CSV + PDF
│       ├── qt_panel.py      # painel principal (PySide6/Qt) — recomendado
│       ├── cv_panel.py      # painel OpenCV (fallback)
│       └── ui.py            # seletor de backend (qt/tk/cv) + painel Tkinter
├── tests/
│   ├── test_detector.py     # + resolve_device, tracking
│   ├── test_posture.py      # sinal de postura de uso de celular
│   ├── test_text_render.py  # renderização de texto Unicode
│   ├── test_temporal.py     # histerese, grace, evicção
│   ├── test_demo.py
│   ├── test_session.py      # dwell time / ocupação
│   ├── test_enrollment.py   # galeria + matrícula + captura (recognizer mockado)
│   ├── test_face_recognizer.py  # match + TTA + cache de identidade
│   ├── test_attendance.py   # presença/identidade/movimentação
│   └── test_reports.py      # geração de CSV/PDF
├── data/
│   ├── sample_video.mp4     # (adicionar manualmente)
│   ├── students/            # fotos de referência + gallery.npz (gitignored)
│   └── reports/             # relatórios gerados (gitignored)
├── requirements.txt
└── README.md
```
