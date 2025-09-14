from diffusers import BitsAndBytesConfig as DF_BNB
from transformers import BitsAndBytesConfig as TF_BNB
import torch

# 4-bit (diffusers / UNet)
unet_qconf = DF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)

# 4-bit (transformers / Text Encoder)
te_qconf = TF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)

# 8-bit (diffusers / UNet)
# Для 8-битного режима достаточно включить load_in_8bit.
# Параметры, специфичные для 4-бит (например, quant_type / compute_dtype), здесь не нужны.
unet_qconf_8bit = DF_BNB(
    load_in_8bit=True,
)

# 8-bit (transformers / Text Encoder)
# Дополнительно можно настраивать порог для int8 (llm_int8_threshold) и offload, если потребуется.
te_qconf_8bit = TF_BNB(
    load_in_8bit=True,
    # llm_int8_threshold=6.0,
    # llm_int8_enable_fp32_cpu_offload=False,
)