"""Эмбеддинги: реализация на SentenceTransformers (CPU/GPU)."""
from typing import List, Protocol
import torch
from dataclasses import dataclass
from src import logger

class Embedder(Protocol):
    """Абстракция энкодера эмбеддингов."""
    name: str
    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]: ...

@dataclass
class STConfig:
    """Конфиг для SentenceTransformers-эмбеддера."""
    model_name: str
    use_half: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class STEmbedder:
    """SentenceTransformers-энкодер: локальный, может работать на CUDA."""
    def __init__(self, cfg: STConfig):
        from sentence_transformers import SentenceTransformer
        self.name = f"st::{cfg.model_name}"
        self.device = cfg.device
        logger.info(f"[embedder] Using {self.name} on {self.device}")
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        if cfg.use_half and self.device.startswith("cuda"):
            try:
                self.model = self.model.to(dtype=torch.float16)
            except Exception:
                pass
    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Преобразовать список текстов в список float-векторов (L2-нормализованных)."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return vecs.tolist()