import os
import queue
import argparse
from typing import Dict, List, Tuple, Any

from config import ModelLLM, ModelLLMInstruct, ModelDiffusers
from src.models.config import ModelConfig
from src.generators.job import llmJob, imageJob
from src.generators.llm import LLMGenerator
from src.generators.diffusers import DiffusersGenerator
from src.api.server import create_app


def _to_model_config(cfg: Any) -> ModelConfig:
    return ModelConfig(
        model_id=getattr(cfg, "model_path", None),
        quant=(getattr(cfg, "quant", None) or "none").lower(),
        dtype=getattr(cfg, "dtype", "bf16"),
        device=getattr(cfg, "device", None),
        local_files_only=bool(getattr(cfg, "local_files_only", True)),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )


# Global queues and job registries
llm_jobs_queue: "queue.Queue[llmJob]" = queue.Queue()
llm_jobs: Dict[str, llmJob] = {}
image_jobs_queue: "queue.Queue[imageJob]" = queue.Queue()
image_jobs: Dict[str, imageJob] = {}


def build_generator_and_app(model_choice: str):
    generators: List[Any] = []

    if model_choice == "images":
        cfg = ModelDiffusers()
        api_mode = "images"
        model_config = _to_model_config(cfg)
        img_gen = DiffusersGenerator(image_jobs_queue, model_config)
        img_gen.start()
        generators.append(img_gen)
        app = create_app(
            llm_jobs_queue=image_jobs_queue,  # reuse for metrics healthz
            llm_jobs=None,
            api_mode=api_mode,
            image_jobs_queue=image_jobs_queue,
            image_jobs=image_jobs,
        )
        return generators, app

    # LLM modes
    if model_choice == "llm-instruct":
        cfg = ModelLLMInstruct()
        api_mode = "chat"
    else:
        cfg = ModelLLM()
        api_mode = "completion"

    model_config = _to_model_config(cfg)

    llm_generator = LLMGenerator(llm_jobs_queue, model_config)
    llm_generator.start()
    generators.append(llm_generator)

    app = create_app(llm_jobs_queue, llm_jobs, api_mode=api_mode)
    return generators, app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["llm", "llm-instruct", "images"],
        default="llm",
        help="Which model type to run: base LLM, instruct (chat) model, or images (Stable Diffusion)",
    )
    args = parser.parse_args()

    generators, flask_app = build_generator_and_app(args.model)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    try:
        flask_app.run(host=host, port=port)
    except KeyboardInterrupt:
        pass
    finally:
        for g in generators:
            try:
                g.stop()
            except Exception:
                pass
            try:
                g.join(timeout=5.0)
            except Exception:
                pass


if __name__ == "__main__":
    main()


