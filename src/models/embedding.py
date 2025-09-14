"""Эмбеддинги: реализация на SentenceTransformers (CPU/GPU)."""
from typing import List, Protocol, Optional
import torch
from dataclasses import dataclass
from src import logger

# Простой маппинг типов для управления dtype при загрузке
_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class embedding_loader:
    """Загрузчик эмбеддинг-моделей (SentenceTransformers) на GPU/CPU.

    Повторяет подход llm_loader (без квантования):
    - model_path: путь к модели (локальный или с HF hub)
    - dtype: один из {"fp16", "fp32", "bf16"} или полные названия (float16/32/bfloat16)
    - device: "cuda" или "cpu" (по умолчанию авто: cuda если доступно)
    - local_files_only / trust_remote_code: пробрасываются при возможности
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bf16",
        device: Optional[str] = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = model_path
        self.dtype = _DTYPE_MAP.get(dtype.lower(), torch.bfloat16)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
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
