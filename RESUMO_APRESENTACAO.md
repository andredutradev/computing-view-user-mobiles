# Computing View — Resumo para Apresentação

> Roteiro enxuto, organizado **slide a slide**, para montar a apresentação em
> PowerPoint. Cada seção abaixo equivale a um slide (título + tópicos).

---

## Slide 1 — Capa

**Computing View — Detecção de Uso de Celular por Visão Computacional**

- Disciplina: Visão Computacional
- Autor: André Dutra
- Sistema que monitora uma sala em tempo real e responde:
  **quem está na sala? quem está no celular? por quanto tempo?**

---

## Slide 2 — O problema / Proposta

- Em salas de aula (ou ambientes de trabalho), o **uso de celular** atrapalha a
  atenção, mas é **difícil de monitorar manualmente**.
- **Proposta:** usar **visão computacional** para detectar automaticamente, a
  partir de uma câmera comum, **quem está usando o celular** — e ainda
  **identificar cada pessoa** e medir **quanto tempo** cada uma ficou no aparelho.
- Entrega final: **relatórios de frequência e de uso de celular** (CSV + PDF).

---

## Slide 3 — Visão geral do sistema (3 pilares)

| Pilar | O que faz |
|-------|-----------|
| **Detecção de uso de celular** | Combina pose (esqueleto), detecção do aparelho e análise de postura para decidir, por pessoa, se está mexendo no celular. |
| **Presença e identidade** | Reconhecimento facial identifica cada aluno, acompanha presença/movimentação e mede o tempo de permanência. |
| **Relatórios** | Frequência por aluno (com tempo de celular) e ocupação da sala ao longo do tempo, em CSV + PDF. |

- Entrada: **webcam ao vivo** ou **arquivo de vídeo**.
- Saída: vídeo anotado em tempo real + relatórios.

---

## Slide 4 — Como funciona (pipeline)

```
Câmera/Vídeo → YOLO (pessoas + esqueleto + celular) → Regras de associação
            → Reconhecimento facial (quem é) → Suavização temporal
            → Vídeo anotado + Relatórios (CSV/PDF)
```

1. Captura o frame (câmera ou vídeo).
2. Detecta **pessoas + esqueleto** e o **celular** com modelos YOLO11.
3. Decide, por pessoa, se está **usando o celular**.
4. **Identifica** cada pessoa por reconhecimento facial.
5. **Estabiliza** a decisão ao longo do tempo (sem piscar).
6. Desenha o resultado na tela e **acumula dados** para o relatório.

---

## Slide 5 — Núcleo de Visão Computacional: dois modelos YOLO

- **Detecção de objetos (YOLO11):** localiza **pessoas** e **celulares**
  (classe `cell phone` do dataset COCO). Cada objeto vira uma *bounding box*.
- **Pose / estimativa de esqueleto (YOLO11-pose):** além da pessoa, extrai os
  **keypoints** do corpo — ombros, cotovelos e **pulsos** (mãos).
- O modelo *medium* + resolução maior (`imgsz 960`) e limiar de confiança
  próprio para o celular ajudam a achar aparelhos **escuros/borrados na mão** —
  o caso real mais difícil.

> Conceitos de VC envolvidos: **detecção de objetos**, **pose estimation**,
> **bounding boxes**, **keypoints**, **rastreamento (tracking)**.

---

## Slide 6 — A decisão "está usando o celular?" (3 sinais)

1. **Proximidade pulso ↔ celular (sinal primário):** a mão (pulso) está perto
   da caixa do celular. O raio se adapta ao tamanho da pessoa → **invariante à
   distância da câmera**.
2. **Postura típica de uso (mesmo sem ver o aparelho):** mão erguida à frente do
   tronco + cotovelo flexionado + **cabeça inclinada para baixo** (olhando a
   tela). Resolve o caso do celular oculto/escuro.
3. **Fallback por sobreposição:** quando os pulsos não estão visíveis, usa-se a
   sobreposição das caixas (IoU + contenção).

- Pessoa **usando celular → caixa verde**; demais → laranja.
- Braços em ciano, mãos em magenta, mão que segura o aparelho em vermelho.

---

## Slide 7 — A matemática por trás (sobreposição de caixas)

Cada caixa é `(x1, y1, x2, y2)` em pixels. Combinamos:

- **Área de interseção** entre as caixas (retângulo de sobreposição).
- **IoU** = interseção / união — métrica clássica de sobreposição.
- **Contenção** = interseção / área do celular — fração do celular dentro da
  pessoa (robusta porque o celular é bem menor que a pessoa).
- **Distância euclidiana** entre centros — desempata qual pessoa "ganha" o
  celular quando há várias candidatas.

> Mostra a ligação direta entre **geometria/álgebra** e a regra de negócio.

---

## Slide 8 — Estabilidade temporal (não piscar)

- Problema: a cada frame a decisão pode oscilar (verde ↔ laranja) com o
  movimento da pessoa.
- Solução de VC: **rastreamento** (ByteTrack/BoT-SORT) dá um `id` estável a cada
  pessoa entre frames + **suavização temporal**:
  - **voto em janela deslizante** das últimas decisões;
  - **histerese** (liga em 0.6, só desliga em 0.35) — elimina o tremido;
  - **grace period** — a caixa sobrevive a oclusões breves.

---

## Slide 9 — Identidade: reconhecimento facial

- **InsightFace** gera um *embedding* (vetor) do rosto de cada pessoa.
- Identidade = **maior similaridade de cosseno** com a galeria de alunos.
- Robusto **a partir de uma única foto**:
  - **augmentation** (espelho horizontal) na matrícula;
  - **Test-Time Augmentation** (média do rosto e do espelho) no reconhecimento;
  - **margem anti-ambiguidade** entre o 1º e o 2º melhor candidato.
- Quem usa o celular sem estar matriculado entra como `Pessoa #ID`.

---

## Slide 10 — Relatórios gerados

| Arquivo | Conteúdo |
|---------|----------|
| `frequencia.csv/pdf` | Por pessoa: presente?, tempo total, **tempo no celular**, **% do tempo no celular**, 1º/último avistamento. |
| `ocupacao.csv/pdf` | Série temporal (pessoas, identificados, usando celular) + pico e média de ocupação. |

- Exportáveis em **CSV + PDF**, prontos para análise.

---

## Slide 11 — Interface e modos de uso

- **Painel desktop (PySide6/Qt):** matrícula com webcam, monitoramento com vídeo
  embutido e geração de relatórios — tudo numa janela.
- **Modo demo** (`--demo`): abre a interface na hora, sem câmera nem download de
  pesos — ótimo para apresentar.
- **Validação por imagens** (`--eval-images`): mede a precisão contra fotos
  reais de pessoas usando o celular.

```bash
python3 -m src.main --demo        # demonstração rápida
python3 -m src.main --ui          # painel completo
python3 -m src.main --attendance  # monitorar com presença
```

---

## Slide 12 — Arquitetura (engenharia de software)

- Princípios **SOLID**, camadas desacopladas por **injeção de dependência**:
  `config → fonte de vídeo → detecção → visualização → orquestração`.
- **Strategy + Factory** para a fonte de vídeo (trocar webcam por arquivo **não**
  altera a lógica principal).
- **Testes** mockam o YOLO e os frames → rodam sem câmera nem download de pesos.

---

## Slide 13 — Relação com a matéria de Visão Computacional

Este projeto aplica, de ponta a ponta, conceitos centrais da disciplina:

- **Detecção de objetos** (YOLO / dataset COCO) — pessoas e celulares.
- **Pose estimation / keypoints** — esqueleto, braços e mãos.
- **Rastreamento de objetos** (tracking multi-objeto entre frames).
- **Reconhecimento facial** (embeddings + similaridade de cosseno).
- **Geometria de imagem** (IoU, contenção, distâncias, ângulos de articulação).
- **Filtragem temporal** (suavização, histerese) para robustez do sinal.
- **Inferência em tempo real** com aceleração de hardware (CUDA / Apple MPS / CPU).

---

## Slide 14 — Considerações de privacidade (LGPD)

- *Embeddings* faciais são **dado pessoal sensível** (LGPD, art. 5º II e art. 11).
- Em produção: **consentimento explícito**, dados biométricos fora de
  repositórios compartilhados, **política de retenção/exclusão**.

---

## Slide 15 — Conclusão

- Sistema **completo e funcional** de visão computacional aplicado a um problema
  real de sala de aula.
- Combina **múltiplas técnicas de VC** (detecção, pose, tracking, reconhecimento
  facial) numa solução em tempo real.
- Entrega valor prático: **identificação, medição de tempo e relatórios**.
- Arquitetura **limpa, testável e extensível**.

**Obrigado!** — Perguntas?
