"""Векторная БД: интерфейс + реализация на ChromaDB."""
from typing import List, Dict, Protocol
from dataclasses import dataclass
from src.core.types import Chunk, QueryHit
import chromadb
from chromadb.config import Settings


class VectorStore(Protocol):
    """Контракт векторного хранилища."""

    def reset(self) -> None: ...

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None: ...

    def query(self, query_vec: List[float], k: int = 6, where: Dict = None) -> List[QueryHit]: ...


@dataclass
class ChromaConfig:
    """Параметры Chroma."""
    persist_dir: str
    collection: str


class ChromaStore:
    """Хранилище на основе ChromaDB (локальное/персистентное)."""

    def __init__(self, cfg: ChromaConfig):
        self.cfg = cfg
        self.client = chromadb.Client(Settings(is_persistent=True, persist_directory=cfg.persist_dir))
        try:
            self.col = self.client.get_collection(cfg.collection)
        except Exception:
            self.col = self.client.create_collection(cfg.collection)

    def reset(self) -> None:
        """Полностью пересоздать коллекцию (удобно для переиндексации)."""
        try:
            self.client.delete_collection(self.cfg.collection)
        except Exception:
            pass
        self.col = self.client.create_collection(self.cfg.collection)

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Добавить пачку записей: (id, text, meta, embedding)."""
        self.col.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.meta for c in chunks],
            embeddings=embeddings
        )

    def query(self, query_vec: List[float], k: int = 6, where: Dict = None) -> List[QueryHit]:
        """Вернуть top-K ближайших чанков к заданному вектору запроса."""
        kwargs = {}
        if where:  # <-- ключевая строка
            kwargs["where"] = where
        res = self.col.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            **kwargs
        )

        hits: List[QueryHit] = []
        docs = res["documents"][0];
        metas = res["metadatas"][0];
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(QueryHit(text=doc, meta=meta, distance=float(dist)))
        return hits