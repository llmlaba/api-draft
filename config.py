from dataclasses import dataclass

@dataclass
class ModelPath:
    llm: str = '/home/sysadmin/llm/mistral'
    tts: str = '/home/sysadmin/llm/bark'
    embeddings: str = '/home/sysadmin/llm/st'
    diffusers: str = '/home/sysadmin/llm/sd1.5'
