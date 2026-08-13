"""
Local dense embeddings for the evidence layer.

Embeddings run ON-MACHINE via fastembed (ONNX runtime, no torch). This is not a
performance choice -- it is a data-handling one. The corpus is real, consented
student coursework whose text must not leave the box (see data/README.md), so
exemplar proof text is never sent to a hosted embedding API. fastembed keeps the
dependency footprint small (a quantized ONNX model, no torch) while the corpus
stays put.

The index is a derived artifact that embeds verbatim student text, so it is
gitignored for the same reason the chunk files are.

What gets embedded
------------------
Only ``graded_exemplar`` chunks are embedded. The 7 ``rubric_guidance`` chunks
are retrieved by direct lookup (see retrieval.py), not vector search -- searching
a 7-document set is less reliable than addressing it by key.

Index format (.npz)
-------------------
    vectors      float32 [N, D]  L2-normalized, so dot product == cosine
    chunk_ids    str    [N]
    question_key str    [N]      access-scoping dimension
    rubric_item  str    [N]      retrieval is scoped to one item at a time
    model        str             model id the vectors were produced with
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import os

import numpy as np

# bge-small-en-v1.5: 384-dim, strong quality/speed trade-off on CPU, and one of
# fastembed's supported models. Override with EVIDENCE_ROOM_EMBED_MODEL (must be
# a fastembed-supported id: see TextEmbedding.list_supported_models()). The index
# records whichever id was used so a query/index model mismatch is detectable
# rather than silently wrong.
DEFAULT_MODEL = os.environ.get("EVIDENCE_ROOM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

DEFAULT_INDEX = Path(__file__).resolve().parent / "embeddings" / "index.npz"


@dataclass
class EmbeddingIndex:
    vectors: np.ndarray          # [N, D] float32, L2-normalized
    chunk_ids: np.ndarray        # [N] str
    question_key: np.ndarray     # [N] str
    rubric_item: np.ndarray      # [N] str
    model: str

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def save(self, path: Path | str = DEFAULT_INDEX) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self.vectors,
            chunk_ids=self.chunk_ids,
            question_key=self.question_key,
            rubric_item=self.rubric_item,
            model=np.array(self.model),
        )
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX) -> "EmbeddingIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No embedding index at {path}. Build it with:\n"
                f"    python -m evidence_room.embeddings --chunks chunks_real.jsonl"
            )
        z = np.load(path, allow_pickle=False)
        return cls(
            vectors=z["vectors"],
            chunk_ids=z["chunk_ids"],
            question_key=z["question_key"],
            rubric_item=z["rubric_item"],
            model=str(z["model"]),
        )


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize so dot product == cosine. fastembed already returns
    normalized vectors for bge, but normalizing defensively keeps the index
    invariant true regardless of the configured model."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


class Embedder:
    """Thin wrapper over a fastembed TextEmbedding model, loaded lazily."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_id = model
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # ONNX import, no torch
            self._model = TextEmbedding(model_name=self.model_id)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64,
            show_progress: bool = False) -> np.ndarray:
        model = self._load()
        embed_iter = model.embed(texts, batch_size=batch_size)
        if show_progress:
            from tqdm import tqdm
            embed_iter = tqdm(embed_iter, total=len(texts), desc="embedding")
        vecs = np.array(list(embed_iter), dtype=np.float32)
        return _l2_normalize(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query proof. bge is an asymmetric model -- queries want a
        retrieval instruction prefix that documents do not. fastembed applies the
        correct prefix internally via ``query_embed``, so we route queries through
        it rather than prefixing by hand.
        """
        model = self._load()
        vec = np.array(list(model.query_embed([text])), dtype=np.float32)
        return _l2_normalize(vec)[0]


def _iter_chunks(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_embedding_index(chunks_path: Path | str,
                          model: str = DEFAULT_MODEL,
                          show_progress: bool = True) -> EmbeddingIndex:
    """
    Embed every ``graded_exemplar`` chunk and return an in-memory index.

    Rubric guidance is intentionally skipped -- it is addressed by key, not
    searched (see retrieval.py / README design notes).
    """
    chunks_path = Path(chunks_path)
    rows = [c for c in _iter_chunks(chunks_path) if c["doc_type"] == "graded_exemplar"]
    if not rows:
        raise ValueError(
            f"No graded_exemplar chunks in {chunks_path}. "
            f"Run ingestion first: python -m evidence_room.ingest"
        )

    texts = [c["text"] for c in rows]
    embedder = Embedder(model)
    vectors = embedder.encode(texts, show_progress=show_progress)

    return EmbeddingIndex(
        vectors=vectors,
        chunk_ids=np.array([c["chunk_id"] for c in rows]),
        question_key=np.array([c["question_key"] or "" for c in rows]),
        rubric_item=np.array([c["provenance"]["rubric_item"] for c in rows]),
        model=model,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the dense embedding index")
    ap.add_argument("--chunks", default="chunks_real.jsonl",
                    help="ingested chunk JSONL (default: chunks_real.jsonl)")
    ap.add_argument("--out", default=str(DEFAULT_INDEX), help="output .npz path")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model id")
    args = ap.parse_args()

    print(f"Embedding graded exemplars from {args.chunks} with {args.model} ...")
    index = build_embedding_index(args.chunks, model=args.model)
    out = index.save(args.out)
    dim = index.vectors.shape[1]
    print(f"Indexed {len(index):,} exemplar chunks | dim={dim} | model={index.model}")
    print(f"Wrote {out}")
