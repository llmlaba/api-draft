"""Diffusion loader: class to load SD 1.5 family pipelines with optional 4/8-bit quantization.
Usage example:
    loader = diffusion_loader(
        model_path="/path/to/stable-diffusion-1-5",
        quant="4bit",            # one of: "none", "4bit", "8bit"
        dtype="bfloat16"          # one of: "float16", "float32", "bfloat16"
    )

"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# Quantization policies for UNet (diffusers) and Text Encoder (transformers)
try:
    from src.quantization.policies import (
        te_qconf,
        te_qconf_8bit,
        unet_qconf,
        unet_qconf_8bit,
    )
except Exception:
    te_qconf = None
    te_qconf_8bit = None
    unet_qconf = None
    unet_qconf_8bit = None


_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class diffusion_loader:
    """Load Stable Diffusion 1.5 family with optional 4/8-bit quantization for TE/UNet.

    Parameters:
    - model_path: local or hub path to the SD model folder
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
        self._pipe: Optional[StableDiffusionPipeline] = None
        self._components: Optional[Tuple[CLIPTextModel, UNet2DConditionModel]] = None

    def _select_quant_confs(self):
        if self.quant == "4bit":
            return te_qconf, unet_qconf
        if self.quant == "8bit":
            return te_qconf_8bit, unet_qconf_8bit
        return None, None

    def load(self) -> StableDiffusionPipeline:
        te_conf, unet_conf = self._select_quant_confs()

        text_encoder = CLIPTextModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder",
            quantization_config=te_conf,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.model_path,
            subfolder="unet",
            quantization_config=unet_conf,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            text_encoder=text_encoder,
            unet=unet,
            torch_dtype=self.dtype,
            safety_checker=None,
            feature_extractor=None,
            use_safetensors=True,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

        try:
            pipe = pipe.to(self.device)
        except Exception:
            pass

        self._components = (text_encoder, unet)
        self._pipe = pipe
        return pipe
