from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_id: str
    dtype: str = 'fp16'
    attn_impl: str = 'sdpa'
    quant: str = 'none'
    low_cpu_mem_usage: bool = True
