"""LLM loader: simple class to load CausalLM models with optional 4/8-bit quantization.
Usage example:
    loader = llm_loader(
        model_path="/path/to/mistral",
        quant="4bit",            # one of: "none", "4bit", "8bit"
        dtype="bfloat16"          # one of: "float16", "float32", "bfloat16"
    )
    model, tokenizer = loader.load()
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

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
    - model_path: local or hub path to model
    - quant: "none" | "4bit" | "8bit" (default: "none")
    - dtype: "float16" | "float32" | "bfloat16" (default: "bfloat16")
    - device: target device, e.g. "cuda" or "cpu" (default: auto: cuda if available)
    - local_files_only: whether to restrict to local files (default: True)
    - trust_remote_code: pass through to HF loaders if custom code is needed
    """

    def __init__(
        self,
        model_path: str,
        quant: str = "none",
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = model_path
        self.quant = (quant or "none").lower()
        self.dtype = _DTYPE_MAP.get(dtype.lower(), torch.bfloat16)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _select_quant_conf(self):
        if self.quant == "4bit":
            return te_qconf
        if self.quant == "8bit":
            return te_qconf_8bit
        return None

    def load(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        qconf = self._select_quant_conf()
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
            pass
        return self._model, self._tokenizer
