"""LLM loader: simple class to load CausalLM models with optional 4/8-bit quantization.
Usage example:
    from src.models.config import ModelConfig

    cfg = ModelConfig(
        model_id="/path/to/mistral",
        quant="4bit",            # one of: "none", "4bit", "8bit"
        dtype="bfloat16",        # one of: "float16", "float32", "bfloat16" / "bf16"
    )
    loader = llm_loader(cfg)
    model, tokenizer = loader.load()
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from src.models.config import ModelConfig
from src.logger import get_logger

# Quantization policies for transformers (text encoder/LLM) live here
try:
    from src.quantization.policies import te_qconf, te_qconf_8bit
except Exception:
    # Fallbacks if policies are unavailable; quantization will be disabled
    te_qconf = None
    te_qconf_8bit = None


_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class llm_loader:
    """Load a CausalLM model with optional 4/8-bit quantization.

    Parameters:
    - config: ModelConfig with fields like model_id, quant, dtype, device,
      local_files_only, trust_remote_code
    """

    def __init__(self, config: ModelConfig) -> None:
        self._log = get_logger(__name__)
        self.config = config
        self.model_path = config.model_id
        self.quant = (config.quant or "none").lower()
        self.dtype = _DTYPE_MAP.get((config.dtype or "bf16").lower(), torch.bfloat16)
        device = config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.local_files_only = bool(config.local_files_only)
        self.trust_remote_code = bool(config.trust_remote_code)
        self._model = None
        self._tokenizer = None

    def _select_quant_conf(self):
        if self.quant == "4bit":
            return te_qconf
        if self.quant == "8bit":
            return te_qconf_8bit
        return None

    def load(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        # Log configuration summary
        self.log.info(
            "[LLM] Loading model",
            extra={
                "model_id": self.model_path,
                "quant": self.quant,
                "dtype": str(self.dtype)
            },
        )
        qconf = self._select_quant_conf()
        if qconf is None and self.quant not in (None, "none"):
            self.log.warning("[LLM] Quantization policy not available; proceeding without quantization", extra={"requested_quant": self.quant})
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=qconf,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        # Move to target device (works for non-quantized; for bnb often still ok)
        try:
            self._model = self._model.to(self.device)
        except Exception:
            self.log.warning("[LLM] Moving model to device failed; using default placement", exc_info=True)
        self.log.info("[LLM] Model and tokenizer loaded")
        return self._model, self._tokenizer
