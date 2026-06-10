"""Matrícula de alunos a partir de fotos de referência (galeria de embeddings).

Layout aceito em ``data/students/``:
  - subpasta por aluno (recomendado, permite várias fotos):
        data/students/ana_silva/frente.jpg
        data/students/ana_silva/lado.jpg
  - OU imagem "plana" (o nome do arquivo vira o id do aluno):
        data/students/bruno_costa.jpg

A galeria é persistida em ``gallery.npz`` (embeddings, compacto) + um sidecar
``gallery.json`` (metadados legíveis: nome, fotos, data, contagem).

Executável de forma headless:
    python3 -m src.attendance.enrollment --photos data/students \
        --out data/students/gallery.npz
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import Config, settings

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Acima desta similaridade de cosseno, dois embeddings são tratados como
# DUPLICATA (ex.: o espelho de uma foto frontal quase simétrica) e o segundo é
# descartado — mantém a galeria enxuta sem perder robustez real.
_DEDUP_COSINE = 0.9995


def slugify(name: str) -> str:
    """Transforma um nome em um id estável (sem acento, minúsculas, _).

    Remove acentos (NFKD) para que "André" e "Andre" gerem o mesmo id, evitando
    duplicar o mesmo aluno por causa de acentuação. O nome de exibição original
    (com acento) é preservado à parte, no ``display_name``.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    return s.strip("_") or "aluno"


def _dedup_embeddings(embeddings: list) -> list:
    """Remove embeddings quase idênticos (cosseno >= ``_DEDUP_COSINE``)."""
    kept: list = []
    for emb in embeddings:
        e = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(e))
        e = e / n if n > 1e-8 else e
        if any(float(np.dot(e, k)) >= _DEDUP_COSINE for k in kept):
            continue
        kept.append(e)
    return kept


def embeddings_from_image(recognizer, image, cfg: Config | None = None):
    """Extrai 1+ embeddings de UMA imagem (com augmentation e gate de qualidade).

    Retorna ``(embeddings, n_faces, det_score)``:
      - ``embeddings``: lista de vetores (float32). Vazia se nenhum rosto válido.
      - ``n_faces``: nº de rostos detectados na imagem (>1 sugere foto ruim).
      - ``det_score``: confiança de detecção do MAIOR rosto.

    Com ``face_enroll_augment``, adiciona o embedding do espelho horizontal —
    o que extrai mais robustez de UMA única foto. Rostos com ``det_score`` abaixo
    de ``face_enroll_min_det_score`` são rejeitados (qualidade).
    """
    cfg = cfg or settings
    faces = recognizer.embed_full_frame(image)
    if not faces:
        return ([], 0, 0.0)
    face = max(faces, key=lambda f: f.area)
    det = float(getattr(face, "det_score", 1.0))
    if det < cfg.face_enroll_min_det_score:
        return ([], len(faces), det)

    embs = [np.asarray(face.embedding, dtype=np.float32)]
    if cfg.face_enroll_augment:
        import cv2

        flipped = recognizer.embed_full_frame(cv2.flip(image, 1))
        if flipped:
            big = max(flipped, key=lambda f: f.area)
            embs.append(np.asarray(big.embedding, dtype=np.float32))
    return (_dedup_embeddings(embs), len(faces), det)


@dataclass(frozen=True)
class StudentRecord:
    """Aluno matriculado: id, nome, K embeddings (K>=1) e proveniência."""

    student_id: str
    display_name: str
    embeddings: np.ndarray  # (K, D) L2-normalizados
    photo_paths: tuple = field(default=())
    enrolled_at: str = ""


class Gallery:
    """Galeria de alunos matriculados (em memória + persistência npz/json)."""

    def __init__(self, records: dict | None = None) -> None:
        self.records: dict = records or {}  # dict[str, StudentRecord]

    # -- consulta -----------------------------------------------------------
    def all_embeddings(self):
        """Devolve ``(matriz (N,D), ids)`` com um id de aluno POR LINHA."""
        mats = []
        ids: list = []
        for sid, rec in self.records.items():
            emb = rec.embeddings
            if emb is None or len(emb) == 0:
                continue
            mats.append(np.asarray(emb, dtype=np.float32))
            ids.extend([sid] * len(emb))
        if not mats:
            return (np.zeros((0, 512), dtype=np.float32), [])
        return (np.vstack(mats).astype(np.float32), ids)

    # -- mutação ------------------------------------------------------------
    def upsert(self, record: StudentRecord) -> None:
        self.records[record.student_id] = record

    def remove(self, student_id: str) -> None:
        self.records.pop(student_id, None)

    def __len__(self) -> int:
        return len(self.records)

    # -- persistência -------------------------------------------------------
    @classmethod
    def load(cls, path) -> "Gallery":
        """Carrega de ``gallery.npz`` (+ sidecar .json). Vazia se não existir."""
        path = Path(path)
        if not path.exists():
            return cls({})
        data = np.load(path, allow_pickle=False)
        matrix = data["embeddings"]
        row_ids = [str(x) for x in data["row_ids"]] if "row_ids" in data else []

        json_path = path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))

        groups = defaultdict(list)
        for i, sid in enumerate(row_ids):
            groups[sid].append(matrix[i])

        records: dict = {}
        for sid, rows in groups.items():
            m = meta.get(sid, {})
            records[sid] = StudentRecord(
                student_id=sid,
                display_name=m.get("display_name", sid),
                embeddings=np.asarray(rows, dtype=np.float32),
                photo_paths=tuple(m.get("photo_paths", [])),
                enrolled_at=m.get("enrolled_at", ""),
            )
        return cls(records)

    def save(self, path) -> Path:
        """Persiste em ``gallery.npz`` + ``gallery.json``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix, ids = self.all_embeddings()
        # row_ids como array de strings unicode (sem pickle).
        np.savez(
            path,
            embeddings=matrix,
            row_ids=np.array(ids if ids else [], dtype="U128"),
        )
        # np.savez garante extensão .npz.
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta = {
            sid: {
                "display_name": r.display_name,
                "photo_paths": list(r.photo_paths),
                "enrolled_at": r.enrolled_at,
                "count": int(len(r.embeddings)),
            }
            for sid, r in self.records.items()
        }
        npz_path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return npz_path


def _discover_students(photos_dir: Path) -> dict:
    """Mapeia ``student_id -> (display_name, [caminhos de foto])``."""
    entries: dict = {}
    if not photos_dir.exists():
        return entries
    for item in sorted(photos_dir.iterdir()):
        if item.is_dir():
            sid = slugify(item.name)
            imgs = [
                p
                for p in sorted(item.iterdir())
                if p.suffix.lower() in IMG_EXTENSIONS
            ]
            entries[sid] = (item.name, imgs)
        elif item.suffix.lower() in IMG_EXTENSIONS:
            sid = slugify(item.stem)
            display, imgs = entries.get(sid, (item.stem, []))
            imgs = list(imgs) + [item]
            entries[sid] = (display, imgs)
    return entries


def enroll_directory(
    photos_dir,
    recognizer,
    gallery_path=None,
    update: bool = True,
    enrolled_at: str = "",
    config: Config | None = None,
):
    """Matricula todos os alunos encontrados em ``photos_dir``.

    Para cada foto, usa ``recognizer.embed_full_frame`` (InsightFace) e guarda o
    embedding do MAIOR rosto. Avisa fotos com 0 ou >1 rostos. Idempotente:
    com ``update=True`` carrega a galeria existente e sobrescreve por aluno.

    Devolve ``(gallery, warnings)``.
    """
    photos_dir = Path(photos_dir)
    gallery = (
        Gallery.load(gallery_path)
        if (gallery_path and update)
        else Gallery({})
    )
    warnings: list = []

    # Import LOCAL do OpenCV apenas quando realmente vamos ler imagens.
    import cv2

    entries = _discover_students(photos_dir)
    if not entries:
        warnings.append(f"Nenhuma foto encontrada em {photos_dir}")

    cfg = config or settings
    for sid, (display, imgs) in entries.items():
        embeddings: list = []
        used_paths: list = []
        for img_path in imgs:
            image = cv2.imread(str(img_path))
            if image is None:
                warnings.append(f"Falha ao ler imagem: {img_path}")
                continue
            embs, n_faces, det = embeddings_from_image(recognizer, image, cfg)
            if n_faces == 0:
                warnings.append(f"0 rostos detectados em {img_path}")
                continue
            if not embs:  # rosto presente, mas reprovado no gate de qualidade
                warnings.append(
                    f"Rosto de baixa qualidade em {img_path} "
                    f"(det_score {det:.2f} < {cfg.face_enroll_min_det_score:.2f}); ignorado"
                )
                continue
            if n_faces > 1:
                warnings.append(
                    f">1 rosto em {img_path}; usando o maior (verifique a foto)"
                )
            embeddings.extend(embs)
            used_paths.append(str(img_path))

        embeddings = _dedup_embeddings(embeddings)
        if embeddings:
            gallery.upsert(
                StudentRecord(
                    student_id=sid,
                    display_name=display,
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                    photo_paths=tuple(used_paths),
                    enrolled_at=enrolled_at,
                )
            )
        else:
            warnings.append(
                f"Aluno '{sid}' não matriculado (nenhuma foto com rosto válido)"
            )

    if gallery_path:
        gallery.save(gallery_path)
    return gallery, warnings


def enroll_capture(
    name: str,
    image: np.ndarray,
    recognizer,
    cfg: Config | None = None,
    enrolled_at: str = "",
):
    """Matricula UM aluno a partir de UMA foto capturada (fluxo da UI).

    Passos:
      1. valida o nome e detecta o rosto (com o gate de qualidade);
      2. salva a foto em ``data/students/<slug>/<timestamp>.jpg`` (vira foto de
         referência persistida, reaproveitável numa rematrícula futura);
      3. ACUMULA os embeddings no aluno existente (ou cria um novo) e persiste a
         galeria.

    Devolve ``(ok, message, gallery)``. ``ok=False`` quando não há rosto válido
    (a UI mostra a mensagem e NÃO grava nada).
    """
    cfg = cfg or settings
    name = (name or "").strip()
    if not name:
        return (False, "Informe o nome do aluno antes de capturar.", None)
    if image is None or getattr(image, "size", 0) == 0:
        return (False, "Quadro de câmera inválido.", None)

    embs, n_faces, det = embeddings_from_image(recognizer, image, cfg)
    if n_faces == 0:
        return (False, "Nenhum rosto detectado na foto. Tente novamente.", None)
    if not embs:
        return (
            False,
            f"Rosto de baixa qualidade (nitidez {det:.2f}). Aproxime-se e "
            "melhore a iluminação.",
            None,
        )
    if n_faces > 1:
        # Não impede, mas avisa: idealmente uma pessoa por foto de matrícula.
        pass

    sid = slugify(name)

    # Salva a foto de referência (import local do cv2 — só aqui escrevemos).
    import cv2

    student_dir = Path(cfg.students_dir) / sid
    student_dir.mkdir(parents=True, exist_ok=True)
    stamp = (enrolled_at or "captura").replace(":", "").replace("-", "").replace("T", "_")
    existing = len(list(student_dir.glob("*.jpg")))
    photo_path = student_dir / f"{stamp}_{existing + 1:02d}.jpg"
    cv2.imwrite(str(photo_path), image)

    # Carrega a galeria e ACUMULA (não sobrescreve) os embeddings do aluno.
    gallery = Gallery.load(cfg.gallery_path)
    prev = gallery.records.get(sid)
    all_embs = list(embs)
    paths = [str(photo_path)]
    if prev is not None and prev.embeddings is not None and len(prev.embeddings):
        all_embs = list(prev.embeddings) + all_embs
        paths = list(prev.photo_paths) + paths
    all_embs = _dedup_embeddings(all_embs)

    gallery.upsert(
        StudentRecord(
            student_id=sid,
            display_name=name,
            embeddings=np.asarray(all_embs, dtype=np.float32),
            photo_paths=tuple(paths),
            enrolled_at=enrolled_at,
        )
    )
    gallery.save(cfg.gallery_path)
    msg = (
        f"'{name}' matriculado(a) — {len(all_embs)} amostra(s) na galeria "
        f"({len(paths)} foto(s))."
    )
    return (True, msg, gallery)


def _main(argv=None) -> int:  # pragma: no cover - entrada headless
    import argparse
    from datetime import datetime

    from src.attendance.face_recognizer import FaceRecognizer
    from src.config import settings

    parser = argparse.ArgumentParser(description="Matrícula de alunos por foto.")
    parser.add_argument("--photos", default=settings.students_dir)
    parser.add_argument("--out", default=settings.gallery_path)
    args = parser.parse_args(argv)

    recognizer = FaceRecognizer(config=settings)
    gallery, warnings = enroll_directory(
        args.photos,
        recognizer,
        gallery_path=args.out,
        enrolled_at=datetime.now().isoformat(timespec="seconds"),
    )
    print(f"[INFO] {len(gallery)} aluno(s) matriculado(s) -> {args.out}")
    for w in warnings:
        print(f"[AVISO] {w}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
