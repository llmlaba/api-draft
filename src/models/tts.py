import torch
import os, numpy as np
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

MODEL_PATH = "/home/sysadmin/llm/bark"  # or "bark-small"

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = BarkModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
    use_safetensors=True,
).to("cuda")

inputs = processor(
    text=["Hi! Am am a dummy robot that speaks with a human voice."],
    voice_preset="v2/en_speaker_6",
    return_tensors="pt",
).to("cuda")

audio = model.generate(
    **inputs,
    do_sample=True,
    fine_temperature=0.4,
    coarse_temperature=0.8,
    pad_token_id = processor.tokenizer.pad_token_id,
)

write_wav(
    "bark_test.wav",
    model.generation_config.sample_rate,
    audio[0].detach().cpu().numpy()
)