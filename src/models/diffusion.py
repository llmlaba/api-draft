from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from diffusers import BitsAndBytesConfig as DF_BNB
from transformers import BitsAndBytesConfig as TF_BNB
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_dir = "/home/sysadmin/llm/sd1.5"

te_qconf = TF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)
text_encoder = CLIPTextModel.from_pretrained(
    model_dir,
    subfolder="text_encoder",
    quantization_config=te_qconf,
    torch_dtype=torch.float16,
)

unet_qconf = DF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    model_dir,
    subfolder="unet",
    quantization_config=unet_qconf,
    torch_dtype=torch.float16,
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    text_encoder=text_encoder,
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    use_safetensors=True,
    local_files_only=True,
)

pipe = pipe.to("cuda")

out = pipe(
    prompt="ford focus 3 on parking, high quality, 8k",
    height=512, width=512, guidance_scale=9, num_inference_steps=80)
out.images[0].save("test_bnb.png", format="PNG")