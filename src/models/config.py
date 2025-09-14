from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Unified configuration for model loaders.

    Fields cover the superset of parameters used across loaders (LLM, diffusion,
    embedding, TTS). Loaders can ignore fields they do not use.
    - model_id: id or path of the model (HF repo id or local path)
    - quant: quantization mode ("none" | "4bit" | "8bit"); used by LLM/diffusion
    - dtype: preferred precision (e.g. "bf16", "fp16", "fp32", "bfloat16", etc.)
    - device: explicit device (e.g. "cuda", "cpu"); if None, auto-select
    - local_files_only: restrict to local cache only
    - trust_remote_code: allow custom code from model repos
    - attn_impl, low_cpu_mem_usage: reserved for advanced loaders/backends
    """
    model_id: str
    quant: str = "none"
    dtype: str = "bf16"
    device: Optional[str] = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    attn_impl: str = "sdpa"
    low_cpu_mem_usage: bool = True
