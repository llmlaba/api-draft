import os
import queue
import argparse
from typing import Dict

from config import ModelLLM, ModelLLMInstruct
from src.models.config import ModelConfig
from src.generators.job import llmJob
from src.generators.llm import LLMGenerator
from src.api.server import create_app


def _to_model_config(cfg: ModelLLM) -> ModelConfig:
    # Normalize dtype aliases and device name
    return ModelConfig(
        model_id=cfg.llm,
        quant=(getattr(cfg, "quant", None) or "none").lower(),
        dtype=cfg.dtype,
        device=cfg.device,
        local_files_only=bool(getattr(cfg, "local_files_only", True)),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )


# Global queue and job registry for LLM tasks
llm_jobs_queue: "queue.Queue[llmJob]" = queue.Queue()
llm_jobs: Dict[str, llmJob] = {}


def build_generator_and_app(model_choice: str):
    # Choose model and API mode
    if model_choice == "llm-instruct":
        cfg = ModelLLMInstruct()
        api_mode = "chat"
    else:
        cfg = ModelLLM()
        api_mode = "completion"

    model_config = _to_model_config(cfg)

    # Initialize and start the LLM generator worker
    llm_generator = LLMGenerator(llm_jobs_queue, model_config)
    llm_generator.start()

    # Create Flask app with the selected API
    app = create_app(llm_jobs_queue, llm_jobs, api_mode=api_mode)
    return llm_generator, app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["llm", "llm-instruct"],
        default="llm",
        help="Which model type to run: base LLM or instruct (chat) model",
    )
    args = parser.parse_args()

    llm_generator, flask_app = build_generator_and_app(args.model)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    try:
        flask_app.run(host=host, port=port)
    except KeyboardInterrupt:
        pass
    finally:
        llm_generator.stop()
        llm_generator.join(timeout=5.0)


if __name__ == "__main__":
    main()


