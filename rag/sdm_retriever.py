"""Local retrieval layer for SDM patient-question support.

This module is intentionally sidecar-only for now: it builds and queries a
local embedding index, but does not alter the live agent flow unless a caller
explicitly asks for retrieval.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


class SDMRetriever:
    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge/sdm",
        index_dir: str | Path = "rag/index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None
        self._records: list[dict] | None = None

    @property
    def ready(self) -> bool:
        return (self.index_dir / "sdm_embeddings.npy").exists() and (
            self.index_dir / "sdm_records.json"
        ).exists()

    def build_index(self) -> None:
        records = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                records.append({"source": str(path), "text": text})
        if not records:
            raise ValueError(f"No SDM knowledge documents found in {self.knowledge_dir}")

        model = self._get_model()
        embeddings = model.encode(
            [r["text"] for r in records],
            normalize_embeddings=True,
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "sdm_embeddings.npy", embeddings)
        (self.index_dir / "sdm_records.json").write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        self._embeddings = np.asarray(embeddings)
        self._records = records

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if not self.ready:
            self.build_index()
        self._load_index()
        assert self._embeddings is not None and self._records is not None
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ q
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(
                source=self._records[i]["source"],
                text=self._records[i]["text"],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def _load_index(self) -> None:
        if self._embeddings is None:
            self._embeddings = np.load(self.index_dir / "sdm_embeddings.npy")
        if self._records is None:
            self._records = json.loads(
                (self.index_dir / "sdm_records.json").read_text(encoding="utf-8")
            )

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
