# Computing View — Documento de Apresentação

> Sistema de **visão computacional** que monitora uma sala (câmera ao vivo ou
> arquivo de vídeo) e responde, em tempo real, a três perguntas:
> **quem está na sala?**, **quem está usando o celular?** e **por quanto tempo?** —
> entregando ao final **relatórios de frequência e de uso de celular** em CSV e PDF.

---

## 1. Visão geral

O projeto nasceu como um detector simples de "pessoa segurando celular" (modelo
YOLO único + sobreposição de caixas). Desde então evoluiu para uma **plataforma
de monitoramento de sala** completa, com três pilares:

| Pilar | O que faz |
|-------|-----------|
| **Detecção de uso de celular** | Combina pose (esqueleto) + detecção do aparelho + análise de postura para decidir, por pessoa, se ela está mexendo no celular. |
| **Presença e identidade** | Reconhecimento facial (InsightFace) identifica cada aluno, acompanha presença/movimentação e contabiliza tempo de permanência (*dwell time*). |
| **Relatórios** | Gera frequência por aluno (com **tempo de celular**) e ocupação da sala ao longo do tempo, exportáveis em CSV + PDF. |

A arquitetura segue **SOLID**, com camadas desacopladas (configuração → fonte de
vídeo → detecção → visualização → orquestração) ligadas por **injeção de
dependência**, o que torna cada peça testável isoladamente.

---

## 2. Como o sistema se comporta

### 2.1. Detecção de uso de celular — três sinais combinados

O coração da decisão "está usando o celular" usa **três sinais** com precedência:

1. **Proximidade pulso ↔ celular (sinal primário).** Com a pose ligada, dois
   modelos YOLO11 rodam juntos:
   - `yolo11s-pose.pt` localiza pessoas **e** o esqueleto (ombros, cotovelos,
     **pulsos**);
   - `yolo11m.pt` localiza o **celular** (classe COCO `cell phone`).

   A pessoa é marcada como "usando" quando um **pulso** cai dentro de um **raio**
   proporcional ao tamanho dela (largura dos ombros) — regra **invariante à
   distância** da câmera. Isso elimina os falsos positivos clássicos (celular
   perto do pé/cintura ou da pessoa ao lado).

2. **Contenção (fallback).** Se **nenhum** pulso confiável está perto, e **apenas**
   para pessoas cujos pulsos não estão visíveis, recai-se na geometria antiga
   (IoU + fração do celular dentro da caixa da pessoa). Se o pulso **é** visível e
   está longe, confiamos nisso: ela **não** está segurando — é aqui que mora o
   ganho de precisão.

3. **Postura (sinal autônomo).** Mesmo **sem** o YOLO ver o aparelho (celular
   escuro/oculto na mão), a **postura típica** de quem mexe no celular já marca o
   uso: mão erguida à frente do tronco + cotovelo flexionado + cabeça inclinada
   para baixo. Cada pista é calculada por geometria pura sobre a pose e combinada
   num *score* `[0..1]`. Isso **generaliza** a detecção para além do que o dataset
   COCO consegue enxergar.

### 2.2. Estabilidade do estado (sem "piscar")

Para o rótulo **não oscilar** (laranja ↔ verde) a cada frame quando a pessoa se
mexe, o sistema:

- **Rastreia** cada pessoa entre frames (**ByteTrack/BoT-SORT**), dando um
  `track_id` estável;
- aplica **suavização temporal com histerese**: vota numa janela deslizante e usa
  **dois limiares** (liga em `0.6`, só desliga em `0.35` — a "zona morta" entre
  eles mata o tremido);
- mantém um **grace period**: o box sobrevive a oclusões/sumiços breves em vez de
  desaparecer.

### 2.3. Feedback visual

- Pessoa **usando celular** → caixa **verde** (com "alvo": cantos em L +
  crosshair); demais → **laranja**.
- **Braços** em ciano, **mãos/pulsos** em magenta; a mão que segura o aparelho
  fica em **vermelho**.
- Alunos identificados aparecem com o **nome**; quem é confirmado só pela postura
  recebe o rótulo `Usando Celular (postura)`.

### 2.4. Adaptação automática de hardware

O dispositivo é escolhido sozinho: **CUDA → MPS (Apple) → CPU**. A inferência
roda em `imgsz=960` por padrão (melhor recall de objetos pequenos como o celular),
e há modo leve para máquinas fracas (`--no-pose`, modelos `nano`, `imgsz=640`).

---

## 3. Funcionalidades por modo de uso

| Comando | Para quê serve |
|---------|----------------|
| `python3 -m src.main` | Webcam em tempo real (modo padrão). |
| `python3 -m src.main --source file --video <arq>` | Processa um arquivo de vídeo. |
| `python3 -m src.main --demo` | **Cena sintética** (sem YOLO/câmera) — abre a janela na hora para demonstrar a lógica. |
| `python3 -m src.main --attendance` | Liga presença + reconhecimento facial + relatórios ao final. |
| `python3 -m src.main --enroll data/students` | **Matrícula**: gera a galeria de embeddings dos alunos a partir de fotos. |
| `python3 -m src.main --ui` | **Painel de controle desktop**: matricular, iniciar câmera e gerar relatório por botões. |
| `python3 -m src.main --eval-images <pasta>` | **Validador**: roda a detecção sobre fotos reais e reporta a taxa de acerto (calibração de limiares sem câmera). |
| `--save out.mp4`, `--no-display`, `--max-frames N` | Gravar a saída anotada, rodar headless (CI) e limitar frames. |

### 3.1. Sistema de presença e relatórios

Ao final de uma sessão com `--attendance`/`--ui`, são gerados em
`data/reports/<fonte>_<timestamp>/`:

- **`frequencia.csv` / `.pdf`** — por aluno: presente?, tempo total, **tempo
  usando o celular**, **% do tempo no celular**, 1º/último avistamento e % da
  sessão. Alunos matriculados nunca vistos aparecem como **ausentes**; quem usa o
  celular sem estar matriculado entra como `Pessoa #ID`.
- **`ocupacao.csv` / `.pdf`** — série temporal (pessoas, identificados, usando
  celular) + pico e média de ocupação.

O tempo é medido em **segundos de vídeo** (`frame / fps`), tornando os relatórios
reproduzíveis a partir de um arquivo.

---

## 4. Arquitetura (camadas)

| Módulo | Responsabilidade |
|--------|------------------|
| `src/config.py` | Configuração global imutável (limiares, classes, cores) — tudo sobrescrevível por variáveis `CVUM_*`. |
| `src/video_source.py` | **Strategy + Factory** para a fonte (webcam/arquivo); expõe o FPS real. |
| `src/detector.py` | Wrappers do YOLO (pose + detecção) + regra de negócio (pulso↔celular, contenção, postura) + tracking. |
| `src/temporal.py` | Suavização temporal/histerese por `track_id` (consistência). |
| `src/visualizer.py` | Desenho de caixas, esqueleto, rótulos e HUD. |
| `src/text_render.py` | Renderização de texto Unicode (acentos) via Pillow/TrueType. |
| `src/eval_images.py` | Validação da detecção contra imagens estáticas. |
| `src/main.py` | Loop principal — orquestra tudo por injeção de dependência. |
| `src/attendance/` | Subsistema de presença: `enrollment`, `face_recognizer`, `geometry`, `attendance`, `session`, `reports`, `ui` (+ `cv_panel`). |

---

## 5. Testes

A suíte tem **85 testes** (`python3 -m pytest -v`). Princípio central: os testes
**mockam o YOLO, o OpenCV e o InsightFace** — **não baixam pesos nem exigem
câmera/GPU**, então rodam rápido em CI.

| Arquivo | Cobertura |
|---------|-----------|
| `test_detector.py` (29) | Geometria pura (IoU, contenção, distância), regra pulso↔celular, `resolve_device`, parsing de pose, tracking. |
| `test_posture.py` (7) | *Score* de postura (mão erguida, cotovelo flexionado, cabeça baixa) e marcação autônoma. |
| `test_temporal.py` (6) | Histerese (liga/desliga), grace period, evicção de tracks. |
| `test_session.py` (8) | *Dwell time*, pontes em sumiços curtos, ocupação. |
| `test_attendance.py` (8) | Presença, identidade por track, movimentação. |
| `test_face_recognizer.py` (6) | *Match* por cosseno + cache de identidade amortizado. |
| `test_enrollment.py` (7) | Galeria de embeddings + matrícula (recognizer mockado). |
| `test_reports.py` (3) | Geração de CSV/PDF. |
| `test_eval_images.py` (5) | Harness de validação por imagens. |
| `test_demo.py` (6) | Cena sintética + detector roteirizado. |

---

## 6. Novas melhorias (ainda não enviadas)

> O commit inicial entregava apenas detecção básica (modelo único + contenção).
> Tudo abaixo é trabalho **local, ainda não commitado/enviado**.

1. **Pipeline de pose (esqueleto).** Decisão por **proximidade do pulso** ao
   celular, em vez da caixa inteira da pessoa — muito mais preciso.
2. **Análise de postura.** Sinal **independente da aparência do aparelho**:
   reconhece a postura de "mexendo no celular" mesmo quando o YOLO não vê o
   telefone. Generaliza além das poucas fotos de calibração.
3. **Rastreamento entre frames** (ByteTrack/BoT-SORT) com `track_id` estável.
4. **Suavização temporal com histerese + grace period** (`temporal.py`) — fim do
   "piscar" do rótulo.
5. **Sistema de presença completo** (`src/attendance/`): reconhecimento facial
   (InsightFace), *dwell time*, movimentação (sentado/movimentando) e relatórios
   CSV + PDF de **frequência** (com tempo de celular) e **ocupação**.
6. **Painel de controle desktop** (`--ui`): matricular, iniciar e gerar relatório
   por botões — com a `Application` rodando em thread trabalhadora.
7. **Validador por imagens** (`--eval-images`): mede a taxa de acerto contra fotos
   reais para calibrar limiares sem abrir a câmera.
8. **Seleção automática de dispositivo** (CUDA → MPS → CPU) e `imgsz` ajustável.
9. **Limiar de confiança separado para o celular** (`CVUM_PHONE_CONF`, padrão
   0.25): recupera o aparelho (que sai com confiança baixa) **sem** afrouxar a
   detecção de pessoas.
10. **FPS real da fonte de vídeo** lido do arquivo — relatórios em segundos
    reprodutíveis.
11. **Configuração 100% por ambiente** (`CVUM_*`) sobre uma `Config` imutável.

---

## 7. Correções de falhas (ainda não enviadas)

1. **Falsos positivos por proximidade de caixa.** Celular perto do pé/cintura, ou
   de uma pessoa ao lado, marcava uso indevido. → Resolvido pela regra
   **pulso↔celular** (e confiança no pulso visível e distante).
2. **Rótulo "piscando" (laranja↔verde).** Oscilava a cada frame com o movimento.
   → Resolvido por **tracking + histerese + grace period**.
3. **Celular escuro/borrado na mão perdido** pelos modelos *nano/small*. →
   Modelo **medium** padrão + `imgsz=960` + limiar de celular próprio + reforço de
   raio pela postura.
4. **Janela preta no macOS.** O Tk do sistema (8.5) abre a janela do Tkinter
   **toda preta**. → O `--ui` detecta e cai num **painel equivalente desenhado
   com OpenCV** (`cv_panel.py`), com botões clicáveis e atalhos de teclado.
   Força-se o backend com `CVUM_UI_BACKEND=cv|tk`.
5. **Acentos do português saindo como "?"/caixas.** O `cv2.putText` (fontes
   Hershey) é só ASCII. → Todo texto agora passa por um **renderizador Pillow/
   TrueType** (`text_render.py`), em **uma única passada** por frame (eficiente).
6. **Auto-instalação do `lap` em runtime.** O ByteTrack/BoT-SORT precisa do `lap`
   para a atribuição linear; sem ele o ultralytics tentava instalar em tempo de
   execução. → Fixado no `requirements.txt`.
7. **Comparação de array NumPy como booleano** no recorte facial (rosto pequeno).
   → Trocado por comparação explícita com `None`.
8. **Robustez de imports pesados.** InsightFace/Pillow/fpdf2/torch usam **import
   local protegido**: quem só usa a geometria (e os testes) não precisa tê-los
   instalados, e os PDFs ainda saem mesmo se o `fpdf2` faltar (CSV garantido).

---

## 8. Privacidade (LGPD)

Os *embeddings* faciais são **dado biométrico sensível** (LGPD, art. 5º II e
art. 11). Antes de produção: obtenha **consentimento explícito**, mantenha
`data/students/` e a galeria fora de repositórios compartilhados (já no
`.gitignore`), defina **política de retenção** e use `Gallery.remove()` para o
direito de eliminação.

---

## 9. Resumo executivo

- **O que faz:** monitora uma sala e detecta, por pessoa, uso de celular,
  presença e tempo de permanência — com relatórios CSV/PDF.
- **Como faz:** três sinais (pulso, contenção, postura) + tracking + suavização +
  reconhecimento facial, em camadas SOLID desacopladas.
- **Qualidade:** 85 testes que não dependem de pesos/câmera; configuração total
  por ambiente; adaptação automática de hardware.
- **Estado:** uma grande evolução **local, pronta para revisão e envio** —
  da detecção básica inicial a uma plataforma de monitoramento completa.
