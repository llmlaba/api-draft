import os
import queue
from typing import Dict

from config import ModelLLM
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

# Build ModelConfig from project-level configuration
_llm_cfg = ModelLLM()
_model_config = _to_model_config(_llm_cfg)

# Initialize and start the LLM generator worker
llm_generator = LLMGenerator(llm_jobs_queue, _model_config)
llm_generator.start()


# Flask application
flask_app = create_app(llm_jobs_queue, llm_jobs)


def main():
    # Run Flask development server by default; can be overridden by WSGI servers
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


