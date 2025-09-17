"""TTS loader: class to load Bark TTS model without quantization.
Usage example:
    from src.models.config import ModelConfig

    cfg = ModelConfig(
        model_id="/path/to/bark",  # e.g. local path or "suno/bark-small"
        dtype="bfloat16"           # one of: "float16", "float32", "bfloat16" / "bf16"
    )
    loader = tts_loader(cfg)
    model, processor = loader.load()
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from transformers import AutoProcessor, BarkModel
from src.models.config import ModelConfig
from src.logger import get_logger


_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class tts_loader:
    """Load a Bark TTS model (no quantization).

    Parameters:
    - model_path: local or hub path to the Bark model
    - dtype: "float16" | "float32" | "bfloat16" (default: "bfloat16")
    - device: target device, e.g. "cuda" or "cpu" (default: auto: cuda if available)
    - local_files_only: whether to restrict to local files (default: True)
    - trust_remote_code: pass through to HF loaders if custom code is needed
    """

    def __init__(self, config: ModelConfig) -> None:
        self._log = get_logger(__name__)
        self.config = config
        self.model_path = config.model_id
        # Accept common dtype aliases; default to bf16
        self.dtype = _DTYPE_MAP.get((config.dtype or "bf16").lower(), torch.bfloat16)
        device = config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.local_files_only = bool(config.local_files_only)
        self.trust_remote_code = bool(config.trust_remote_code)
        self._model: Optional[BarkModel] = None
        self._processor: Optional[AutoProcessor] = None

    def load(self) -> Tuple[BarkModel, AutoProcessor]:
        self._log.info(
            "[TTS] Loading Bark model",
            extra={
                "model_id": self.model_path,
                "dtype": str(self.dtype)
            },
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = BarkModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            use_safetensors=True,
            trust_remote_code=self.trust_remote_code,
        )
        try:
            self._model = self._model.to(self.device)
        except Exception:
            self._log.warning("[TTS] Moving model to device failed; using default placement", exc_info=True)
        self._log.info("[TTS] Model and processor loaded")
        return self._model, self._processor