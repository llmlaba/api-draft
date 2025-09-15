from dataclasses import dataclass

@dataclass
class ModelLLM:
    model_path: str = '/home/sysadmin/llm/mistral'
    quant: str = "none"
    dtype: str = "fp16"
    device: str = "cuda"
    local_files_only: bool = True
    trust_remote_code: bool = True

class ModelLLMInstruct:
    model_path: str = '/home/sysadmin/llm/mistral-instruct'
    quant: str = "none"
    dtype: str = "fp16"
    device: str = "cuda"
    local_files_only: bool = True
    trust_remote_code: bool = True

class ModelDiffusers:
    model_path: str = '/home/sysadmin/llm/sd1.5'
    quant: str = "none"
    dtype: str = "fp16"
    device: str = "cuda"
    local_files_only: bool = True
    trust_remote_code: bool = True

class ModelTTS:
    tts: str = '/home/sysadmin/llm/bark'

class ModelEmbedding:
    model_path: str = '/home/sysadmin/llm/st'
