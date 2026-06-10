"""Subsistema de presença em sala de aula.

Reúne, sobre a pipeline de detecção/rastreamento já existente:
  - ``enrollment``      : matrícula de alunos a partir de fotos de referência.
  - ``face_recognizer`` : reconhecimento facial (InsightFace) + cache por track.
  - ``geometry``        : recorte da cabeça (ROI) e similaridade de cosseno.
  - ``attendance``      : presença/movimentação em tempo real por track_id.
  - ``session``         : modelo de dados (dwell time, ocupação) da sessão.
  - ``reports``         : geração de relatórios CSV + PDF baixáveis.
  - ``ui``              : painel de controle desktop (Tkinter) opcional.

Imports pesados (insightface, onnxruntime, fpdf, tkinter) são feitos de forma
LOCAL dentro de cada componente, para que importar o pacote não exija nada além
da stdlib + numpy.
"""
