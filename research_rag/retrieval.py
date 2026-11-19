
"""BGE embeddings in persistent Chroma and in-memory BM25 with RRF."""

from __future__ import annotations

from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .core import Passage, tokenize, rrf


class Retriever:
    def __init__(
        self,
        db_dir: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "research_papers",
    ):
        self.model_name = embedding_model
        self.collection_name = collection_name

        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=str(db_dir))

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.model_name,
            },
        )

        self.chunks: dict[str, Passage] = {}
        self.bm25 = None
        self._load_lexical()

    def _load_lexical(self):
        self.chunks = {}
        offset = 0

        while True:
            result = self.collection.get(
                limit=500,
                offset=offset,
                include=["documents", "metadatas"],
            )

            ids = result["ids"]

            for i, pid in enumerate(ids):
                meta = result["metadatas"][i]

                self.chunks[pid] = Passage(
                    pid,
                    result["documents"][i],
                    meta["source"],
                    int(meta["page"]),
                    meta["doc_id"],
                )

            offset += len(ids)

            if len(ids) < 500:
                break

        self.ordered = list(self.chunks.values())

        self.bm25 = (
            BM25Okapi([tokenize(p.text) for p in self.ordered])
            if self.ordered
            else None
        )

    def index(
        self,
        passages: list[Passage],
        batch_size: int = 64,
        replace: bool = False,
    ) -> int:

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if replace:
            self.client.delete_collection(
                name=self.collection_name
            )

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": self.model_name,
                },
            )

        if not passages:
            self._load_lexical()
            return 0

        old_model = (
            self.collection.metadata.get("embedding_model")
            if self.collection.metadata
            else None
        )

        if old_model and old_model != self.model_name:
            raise ValueError(
                "Embedding model changed; reindex with --replace"
            )

        existing = set(
            self.collection.get(include=[])["ids"]
        )

        new = [
            p for p in passages
            if p.id not in existing
        ]

        for start in range(0, len(new), batch_size):
            batch = new[start:start + batch_size]

            vectors = self.embedder.encode(
                [p.text for p in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            self.collection.upsert(
                ids=[p.id for p in batch],
                embeddings=vectors,
                documents=[p.text for p in batch],
                metadatas=[p.metadata() for p in batch],
            )

        self._load_lexical()
        return len(new)

    def semantic(self, query: str, n: int) -> list[str]:
        if not self.chunks or n <= 0:
            return []

        q = self.embedder.encode(
            [
                "Represent this sentence for searching relevant passages: "
                + query
            ],
            normalize_embeddings=True,
        ).tolist()

        result = self.collection.query(
            query_embeddings=q,
            n_results=min(n, len(self.chunks)),
            include=["distances"],
        )

        return result["ids"][0]

    def lexical(self, query: str, n: int) -> list[str]:
        terms = tokenize(query)

        if self.bm25 is None or not terms or n <= 0:
            return []

        scores = self.bm25.get_scores(terms)

        indices = sorted(
            range(len(scores)),
            key=lambda i: -scores[i],
        )

        return [
            self.ordered[i].id
            for i in indices[:n]
            if scores[i] > 0
        ]

    def search(
        self,
        queries: list[str],
        n_each: int = 15,
        n_fused: int = 15,
        mode: str = "hybrid",
    ) -> list[Passage]:

        if mode not in ("hybrid", "semantic", "bm25"):
            raise ValueError(f"Unknown retrieval mode: {mode}")

        lists = []

        for q in dict.fromkeys(queries):
            if mode in ("hybrid", "semantic"):
                lists.append(self.semantic(q, n_each))

            if mode in ("hybrid", "bm25"):
                lists.append(self.lexical(q, n_each))

        return [
            self.chunks[pid]
            for pid, _ in rrf(lists)[:n_fused]
        ]


class Reranker:
    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model)

    def rank(
        self,
        query: str,
        passages: list[Passage],
        n: int = 6,
    ) -> list[Passage]:

        if not passages or n <= 0:
            return []

        scores = self.model.predict(
            [(query, p.text) for p in passages]
        )

        indices = sorted(
            range(len(passages)),
            key=lambda i: -float(scores[i]),
        )

        return [
            passages[i]
            for i in indices[:n]
        ]
