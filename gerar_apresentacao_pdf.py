"""Gera o PDF de apresentação do Computing View — conteúdo montado do zero.

Não converte Markdown: o conteúdo e o layout são definidos AQUI, usando fpdf2
e as fontes DejaVu empacotadas (Unicode, acentos do português).

Uso: python3 gerar_apresentacao_pdf.py [saida.pdf]   (padrão: APRESENTACAO.pdf)
"""

from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

BASE = Path(__file__).resolve().parent
FONT_REG = BASE / "assets" / "fonts" / "DejaVuSans.ttf"
FONT_BOLD = BASE / "assets" / "fonts" / "DejaVuSans-Bold.ttf"

# Paleta
INK = (33, 37, 41)
ACCENT = (22, 105, 122)        # verde-azulado (tema "visão")
ACCENT_SOFT = (224, 240, 243)
MUTED = (110, 110, 110)
RULE = (210, 210, 210)
QUOTE_BG = (244, 246, 248)
HEAD_BG = (22, 105, 122)
ROW_ALT = (247, 250, 251)


class Apresentacao(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*MUTED)
        self.cell(
            0, 8,
            f"Computing View — Apresentação do Sistema      ·      pág. {self.page_no()}",
            align="C",
        )


def make() -> Apresentacao:
    pdf = Apresentacao(orientation="P", unit="mm", format="A4")
    pdf.add_font("DejaVu", "", str(FONT_REG))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    return pdf


# ---------------------------------------------------------------------------
# Blocos de layout
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    """Escapa o ``--`` (flags de CLI) para o markdown do fpdf2 não o tratar
    como sublinhado. Mantém ``**negrito**`` intacto."""
    return text.replace("--", "\\--")


def h1(pdf, text):
    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(pdf.epw, 8, text)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.5)
    y = pdf.get_y() + 0.5
    pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
    pdf.set_line_width(0.2)
    pdf.ln(3.5)
    pdf.set_text_color(*INK)


def h2(pdf, text):
    pdf.ln(1.5)
    pdf.set_font("DejaVu", "B", 11.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw, 6, text)
    pdf.ln(1.5)


def para(pdf, text, size=10.3, lh=5.4, gap=2.2):
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw, lh, _esc(text), markdown=True)
    pdf.ln(gap)


def bullets(pdf, items, ordered=False, size=10.3, lh=5.3):
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(*INK)
    for idx, it in enumerate(items, 1):
        marker = f"{idx}." if ordered else "•"
        x0 = pdf.l_margin + 2
        pdf.set_x(x0)
        pdf.set_font("DejaVu", "B" if ordered else "", size)
        pdf.cell(7, lh, marker)
        pdf.set_font("DejaVu", "", size)
        pdf.set_x(x0 + 7)
        pdf.multi_cell(pdf.epw - 9, lh, _esc(it), markdown=True)
        pdf.ln(0.6)
    pdf.ln(1.8)


def quote(pdf, text):
    pdf.set_font("DejaVu", "", 10)
    pdf.set_fill_color(*QUOTE_BG)
    pdf.set_text_color(70, 80, 90)
    x0 = pdf.get_x()
    start_y = pdf.get_y()
    pdf.set_x(x0 + 3.5)
    pdf.multi_cell(pdf.epw - 3.5, 5.2, _esc(text), fill=True, markdown=True)
    end_y = pdf.get_y()
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.9)
    pdf.line(x0 + 0.6, start_y, x0 + 0.6, end_y)
    pdf.set_line_width(0.2)
    pdf.set_text_color(*INK)
    pdf.ln(3)


def table(pdf, headers, rows, col_widths, font_size=8.7):
    head_face = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=HEAD_BG)
    pdf.set_font("DejaVu", "", font_size)
    with pdf.table(
        markdown=True,
        line_height=4.9,
        text_align="LEFT",
        col_widths=col_widths,
        headings_style=head_face,
        cell_fill_color=ROW_ALT,
        cell_fill_mode="ROWS",
        borders_layout="MINIMAL",
        width=pdf.epw,
    ) as t:
        hr = t.row()
        for h in headers:
            hr.cell(_esc(h))
        for row in rows:
            r = t.row()
            for c in row:
                r.cell(_esc(c))
    pdf.ln(3)


# ---------------------------------------------------------------------------
# Conteúdo
# ---------------------------------------------------------------------------
def build(pdf: Apresentacao) -> None:
    # ---- Capa ----
    pdf.add_page()
    pdf.ln(28)
    pdf.set_fill_color(*ACCENT)
    pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, 0.0)
    pdf.set_font("DejaVu", "B", 30)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(pdf.epw, 13, "Computing View",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "B", 15)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw, 8, "Documento de Apresentação do Sistema",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.6)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + 60, y)
    pdf.set_line_width(0.2)
    pdf.ln(7)
    pdf.set_font("DejaVu", "", 11.5)
    pdf.set_text_color(70, 80, 90)
    pdf.multi_cell(
        pdf.epw, 6.5,
        "Sistema de visão computacional que monitora uma sala (câmera ao vivo "
        "ou arquivo de vídeo) e responde, em tempo real, a três perguntas: "
        "quem está na sala?, quem está usando o celular? e por quanto tempo? — "
        "entregando ao final relatórios de frequência e de uso de celular em "
        "CSV e PDF.",
    )
    pdf.ln(10)
    # cartões-resumo dos pilares
    pdf.set_font("DejaVu", "B", 11)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(pdf.epw, 6, "Três pilares")
    pdf.ln(1)
    bullets(pdf, [
        "**Detecção de uso de celular** — pose + detecção do aparelho + postura.",
        "**Presença e identidade** — reconhecimento facial e tempo de permanência.",
        "**Relatórios** — frequência por aluno e ocupação da sala (CSV + PDF).",
    ])
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        pdf.epw, 5,
        "Arquitetura em camadas SOLID, ligadas por injeção de dependência. "
        "85 testes automatizados que não dependem de pesos YOLO nem de câmera.",
    )

    # ---- 1. Como o sistema se comporta ----
    pdf.add_page()
    h1(pdf, "1. Como o sistema se comporta")

    h2(pdf, "1.1. Detecção de uso de celular — três sinais combinados")
    para(pdf, "A decisão \"está usando o celular\" usa três sinais, com precedência:")
    bullets(pdf, [
        "**Proximidade pulso ↔ celular (primário).** Dois modelos YOLO11 rodam "
        "juntos: a pose localiza pessoas e o esqueleto (ombros, cotovelos, "
        "**pulsos**) e a detecção localiza o **celular**. A pessoa é marcada "
        "quando um pulso cai dentro de um raio proporcional ao seu tamanho — "
        "regra invariante à distância da câmera. Elimina os falsos positivos "
        "(celular perto do pé/cintura ou de quem está ao lado).",
        "**Contenção (fallback).** Só quando nenhum pulso confiável está perto, "
        "e apenas para pessoas com pulsos não visíveis, usa-se a geometria "
        "antiga (IoU + fração do celular dentro da pessoa). Pulso visível e "
        "longe significa \"não está segurando\" — é o ganho de precisão.",
        "**Postura (autônomo).** Mesmo sem o YOLO ver o aparelho (celular escuro/"
        "oculto), a postura típica já marca o uso: mão erguida à frente do "
        "tronco + cotovelo flexionado + cabeça inclinada para baixo. Generaliza "
        "além do que o dataset COCO enxerga.",
    ], ordered=True)

    h2(pdf, "1.2. Estabilidade do estado (sem \"piscar\")")
    para(pdf, "Para o rótulo não oscilar (laranja ↔ verde) a cada frame quando a "
              "pessoa se mexe, o sistema:")
    bullets(pdf, [
        "**Rastreia** cada pessoa entre frames (**ByteTrack/BoT-SORT**), com "
        "track_id estável;",
        "aplica **suavização temporal com histerese** — vota numa janela "
        "deslizante e usa dois limiares (liga em 0.6, só desliga em 0.35; a "
        "\"zona morta\" entre eles mata o tremido);",
        "mantém um **grace period**: o box sobrevive a oclusões/sumiços breves.",
    ])

    h2(pdf, "1.3. Feedback visual e adaptação de hardware")
    bullets(pdf, [
        "Pessoa usando celular → caixa **verde** (com alvo/crosshair); demais → "
        "**laranja**. Braços em ciano, mãos em magenta; a mão que segura o "
        "aparelho em vermelho.",
        "Alunos identificados exibem o **nome**; uso confirmado só pela postura "
        "recebe o rótulo \"Usando Celular (postura)\".",
        "O dispositivo é escolhido sozinho: **CUDA → MPS (Apple) → CPU**. "
        "Inferência em imgsz=960 por padrão; modo leve para máquinas fracas.",
    ])

    # ---- 2. Funcionalidades por modo de uso ----
    pdf.add_page()
    h1(pdf, "2. Funcionalidades por modo de uso")
    table(
        pdf,
        ["Comando", "Para quê serve"],
        [
            ["python3 -m src.main", "Webcam em tempo real (modo padrão)."],
            ["--source file --video <arq>", "Processa um arquivo de vídeo."],
            ["--demo", "Cena sintética (sem YOLO/câmera) — abre a janela na hora."],
            ["--attendance", "Liga presença + reconhecimento facial + relatórios."],
            ["--enroll data/students", "Matrícula: gera a galeria de embeddings dos alunos."],
            ["--ui", "Painel de controle desktop (matricular, iniciar, relatório)."],
            ["--eval-images <pasta>", "Valida a detecção em fotos reais (calibra limiares)."],
            ["--save / --no-display / --max-frames", "Gravar saída anotada, rodar headless, limitar frames."],
        ],
        col_widths=(42, 58),
    )

    h2(pdf, "2.1. Sistema de presença e relatórios")
    para(pdf, "Ao final de uma sessão com --attendance/--ui, são gerados em "
              "data/reports/<fonte>_<timestamp>/:")
    table(
        pdf,
        ["Arquivo", "Conteúdo"],
        [
            ["frequencia.csv / .pdf",
             "Por aluno: presente?, tempo total, **tempo usando o celular**, % "
             "do tempo no celular, 1º/último avistamento, % da sessão. Ausentes "
             "aparecem explicitamente; anônimos que usam o celular entram como "
             "\"Pessoa #ID\"."],
            ["ocupacao.csv / .pdf",
             "Série temporal (pessoas, identificados, usando celular) + pico e "
             "média de ocupação."],
        ],
        col_widths=(30, 70),
    )
    quote(pdf, "O tempo é medido em segundos de vídeo (frame / fps), tornando os "
               "relatórios reproduzíveis a partir de um arquivo.")

    # ---- 3. Arquitetura ----
    pdf.add_page()
    h1(pdf, "3. Arquitetura (camadas)")
    table(
        pdf,
        ["Módulo", "Responsabilidade"],
        [
            ["config.py", "Configuração global imutável (limiares, classes, cores); tudo via CVUM_*."],
            ["video_source.py", "Strategy + Factory para a fonte (webcam/arquivo); expõe o FPS real."],
            ["detector.py", "YOLO (pose + detecção) + regra de negócio (pulso, contenção, postura) + tracking."],
            ["temporal.py", "Suavização temporal/histerese por track_id."],
            ["visualizer.py", "Desenho de caixas, esqueleto, rótulos e HUD."],
            ["text_render.py", "Texto Unicode (acentos) via Pillow/TrueType."],
            ["eval_images.py", "Validação da detecção contra imagens estáticas."],
            ["main.py", "Loop principal — orquestra tudo por injeção de dependência."],
            ["attendance/", "Presença: enrollment, face_recognizer, geometry, attendance, session, reports, ui."],
        ],
        col_widths=(30, 70),
    )

    # ---- 4. Testes ----
    h1(pdf, "4. Testes")
    para(pdf, "Suíte com **85 testes** (python3 -m pytest -v). Princípio central: "
              "os testes **mockam YOLO, OpenCV e InsightFace** — não baixam pesos "
              "nem exigem câmera/GPU, então rodam rápido em CI.")
    table(
        pdf,
        ["Arquivo", "Qtd.", "Cobertura"],
        [
            ["test_detector.py", "29", "Geometria, regra pulso↔celular, device, parsing de pose, tracking."],
            ["test_posture.py", "7", "Score de postura e marcação autônoma."],
            ["test_temporal.py", "6", "Histerese, grace period, evicção de tracks."],
            ["test_session.py", "8", "Dwell time, pontes em sumiços curtos, ocupação."],
            ["test_attendance.py", "8", "Presença, identidade por track, movimentação."],
            ["test_face_recognizer.py", "6", "Match por cosseno + cache de identidade."],
            ["test_enrollment.py", "7", "Galeria de embeddings + matrícula."],
            ["test_reports.py", "3", "Geração de CSV/PDF."],
            ["test_eval_images.py", "5", "Harness de validação por imagens."],
            ["test_demo.py", "6", "Cena sintética + detector roteirizado."],
        ],
        col_widths=(34, 10, 56),
    )

    # ---- 5. Novas melhorias ----
    pdf.add_page()
    h1(pdf, "5. Novas melhorias (ainda não enviadas)")
    quote(pdf, "O commit inicial entregava apenas detecção básica (modelo único + "
               "contenção). Tudo abaixo é trabalho local, ainda não enviado.")
    bullets(pdf, [
        "**Pipeline de pose (esqueleto)** — decisão por proximidade do pulso, "
        "muito mais precisa que a caixa inteira.",
        "**Análise de postura** — sinal independente da aparência do aparelho.",
        "**Rastreamento entre frames** (ByteTrack/BoT-SORT) com track_id estável.",
        "**Suavização temporal com histerese + grace period** — fim do \"piscar\".",
        "**Sistema de presença completo** — reconhecimento facial, dwell time, "
        "movimentação e relatórios CSV + PDF (frequência e ocupação).",
        "**Painel de controle desktop** (--ui) com a Application em thread.",
        "**Validador por imagens** (--eval-images) para calibrar sem câmera.",
        "**Seleção automática de dispositivo** (CUDA → MPS → CPU) e imgsz ajustável.",
        "**Limiar de confiança separado para o celular** (CVUM_PHONE_CONF) — "
        "recupera o aparelho sem afrouxar a detecção de pessoas.",
        "**FPS real da fonte de vídeo** — relatórios em segundos reprodutíveis.",
        "**Configuração 100% por ambiente** (CVUM_*) sobre uma Config imutável.",
    ], ordered=True)

    # ---- 6. Correções de falhas ----
    h1(pdf, "6. Correções de falhas (ainda não enviadas)")
    bullets(pdf, [
        "**Falsos positivos por proximidade de caixa** (celular perto do pé/"
        "cintura ou de quem está ao lado) → regra pulso↔celular.",
        "**Rótulo \"piscando\"** (laranja↔verde) → tracking + histerese + grace.",
        "**Celular escuro/borrado perdido** pelos modelos nano/small → modelo "
        "medium + imgsz=960 + limiar próprio + reforço de raio pela postura.",
        "**Janela preta no macOS** (Tk 8.5) → painel equivalente em OpenCV "
        "(cv_panel.py), com botões e atalhos; backend via CVUM_UI_BACKEND.",
        "**Acentos saindo como \"?\"/caixas** (cv2.putText é só ASCII) → texto "
        "via Pillow/TrueType, em uma única passada por frame.",
        "**Auto-instalação do lap em runtime** (ByteTrack) → fixado no requirements.txt.",
        "**Comparação de array NumPy como booleano** no recorte facial → "
        "comparação explícita com None.",
        "**Imports pesados protegidos** (InsightFace/Pillow/fpdf2/torch) — testes "
        "e geometria não exigem tê-los; CSV sai mesmo sem fpdf2.",
    ], ordered=True)

    # ---- 7. LGPD + resumo ----
    pdf.add_page()
    h1(pdf, "7. Privacidade (LGPD)")
    para(pdf, "Os embeddings faciais são dado biométrico sensível (LGPD, art. 5º "
              "II e art. 11). Antes de produção: obtenha consentimento explícito, "
              "mantenha data/students/ e a galeria fora de repositórios "
              "compartilhados (já no .gitignore), defina política de retenção e "
              "use Gallery.remove() para o direito de eliminação.")

    h1(pdf, "8. Resumo executivo")
    bullets(pdf, [
        "**O que faz:** monitora a sala e detecta, por pessoa, uso de celular, "
        "presença e tempo de permanência — com relatórios CSV/PDF.",
        "**Como faz:** três sinais (pulso, contenção, postura) + tracking + "
        "suavização + reconhecimento facial, em camadas SOLID desacopladas.",
        "**Qualidade:** 85 testes sem dependência de pesos/câmera; configuração "
        "total por ambiente; adaptação automática de hardware.",
        "**Estado:** grande evolução local, pronta para revisão e envio — da "
        "detecção básica a uma plataforma de monitoramento completa.",
    ])


def main(argv):
    out = Path(argv[1]) if len(argv) > 1 else BASE / "APRESENTACAO.pdf"
    pdf = make()
    build(pdf)
    pdf.output(str(out))
    print(f"[OK] PDF gerado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
