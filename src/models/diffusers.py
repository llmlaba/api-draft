"""Diffusion loader: class to load SD 1.5 family pipelines with optional 4/8-bit quantization.
Usage example:
    from src.models.config import ModelConfig

    cfg = ModelConfig(
        model_id="/path/to/stable-diffusion-1-5",
        quant="4bit",            # one of: "none", "4bit", "8bit"
        dtype="bfloat16"         # one of: "float16", "float32", "bfloat16" / "bf16"
    )
    loader = diffusion_loader(cfg)

"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from src.models.config import ModelConfig
from src.logger import get_logger

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
        self._pipe: Optional[StableDiffusionPipeline] = None
        self._components: Optional[Tuple[CLIPTextModel, UNet2DConditionModel]] = None

    def _select_quant_confs(self):
        if self.quant == "4bit":
            return te_qconf, unet_qconf
        if self.quant == "8bit":
            return te_qconf_8bit, unet_qconf_8bit
        return None, None

    def load(self) -> StableDiffusionPipeline:
        self._log.info(
            "[SD] Loading Stable Diffusion pipeline",
            extra={
                "model_id": self.model_path,
                "quant": self.quant,
                "dtype": str(self.dtype)
            },
        )
        te_conf, unet_conf = self._select_quant_confs()
        if (te_conf is None or unet_conf is None) and self.quant not in (None, "none"):
            self._log.warning("[SD] Quantization policies not available; proceeding without quantization", extra={"requested_quant": self.quant})

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
            self._log.warning("[SD] Moving pipeline to device failed; using default placement", exc_info=True)

        self._components = (text_encoder, unet)
        self._pipe = pipe
        self._log.info("[SD] Pipeline loaded")
        return pipe
