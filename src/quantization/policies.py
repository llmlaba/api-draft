from diffusers import BitsAndBytesConfig as DF_BNB
from transformers import BitsAndBytesConfig as TF_BNB

unet_qconf = DF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)

te_qconf = TF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)
