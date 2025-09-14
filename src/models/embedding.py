"""Эмбеддинги: реализация на SentenceTransformers (CPU/GPU).

Пример использования:
    from src.models.config import ModelConfig

    cfg = ModelConfig(
        model_id="intfloat/e5-small",
        dtype="bf16",
    )
    loader = embedding_loader(cfg)
"""
from typing import List, Protocol, Optional
import torch
from dataclasses import dataclass
from src import logger
from src.models.config import ModelConfig

# Простой маппинг типов для управления dtype при загрузке
_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class embedding_loader:
    """Загрузчик эмбеддинг-моделей (SentenceTransformers) на GPU/CPU.

    Повторяет подход llm_loader (без квантования). Параметры:
    - config: ModelConfig (model_id, dtype, device, local_files_only, trust_remote_code)
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model_path = config.model_id
        self.dtype = _DTYPE_MAP.get((config.dtype or "bf16").lower(), torch.bfloat16)
        device = config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.local_files_only = bool(config.local_files_only)
        self.trust_remote_code = bool(config.trust_remote_code)
        self._model = None

    def load(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"[embedder] Loading {self.model_path} on {self.device} (dtype={self.dtype})")
        # Разные версии SentenceTransformers поддерживают разные параметры конструктора.
        # Пробуем с максимальным набором, затем деградируем.
        try:
            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=self.trust_remote_code,  # может не поддерживаться в старых версиях
                local_files_only=self.local_files_only,    # может не поддерживаться
            )
        except TypeError:
            try:
                self._model = SentenceTransformer(
                    self.model_path,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                )
            except TypeError:
                self._model = SentenceTransformer(self.model_path, device=self.device)
        # Попытаться привести вес к нужному dtype (актуально для CUDA)
        if isinstance(self.dtype, torch.dtype) and str(self.device).startswith("cuda"):
            try:
                self._model = self._model.to(dtype=self.dtype)
            except Exception:
                pass
        return self._model
