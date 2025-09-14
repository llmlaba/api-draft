import argparse

def main(argv=None):

    ap = argparse.ArgumentParser(prog='pt-llm-bench', description='PyTorch LLM bench (llama-bench-like)')
    # possible models llm, tss, diffusers, asm, embedder
    ap.add_argument('-m','--models', default=['llm'])
    ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16','bf16','fp32'])
    ap.add_argument('--quant', type=str, default='none', choices=['none','4bit'])
    ap.add_argument('--attn', type=str, default='sdpa')
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        console.print('[yellow]WARNING[/]: CUDA/ROCm device not available; running on CPU will be slow.')

    model_cfg = ModelConfig(model_id=args.models, dtype=args.dtype, attn_impl=args.attn, quant=args.quant)

    backend = detect_backend(); device = device_string()
    render_header(console, torch.__version__, _tf.__version__, backend, device)

    loader = ModelLoader(model_cfg)

    return 0