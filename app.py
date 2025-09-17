import os
import queue
import argparse
from typing import Dict, List, Any, Set

from config import ModelLLM, ModelLLMInstruct, ModelDiffusers, ModelTTS, LOG_LEVEL
from src.models.config import ModelConfig
from src.generators.job import llmJob, imageJob, speechJob
from src.generators.llm import LLMGenerator
from src.generators.diffusers import DiffusersGenerator
from src.generators.speech import SpeechGenerator
from src.api.server import create_app, Deps
from src.logger import set_global_log_level, get_logger

# Global queues and job registries
# LLM / Chat
llm_jobs_queue: "queue.Queue[llmJob]" = queue.Queue()
llm_jobs: Dict[str, llmJob] = {}
# Images (Stable Diffusion)
image_jobs_queue: "queue.Queue[imageJob]" = queue.Queue()
image_jobs: Dict[str, imageJob] = {}
# Speech (TTS) Bark
speech_jobs_queue: "queue.Queue[speechJob]" = queue.Queue()
speech_jobs: Dict[str, speechJob] = {}
# Reserved for future ASR
audio_jobs_queue: "queue.Queue[Any]" = queue.Queue()
audio_jobs: Dict[str, Any] = {}


def _to_model_config(cfg: Any) -> ModelConfig:
    # Map various config classes to ModelConfig
    model_id = getattr(cfg, "model_path", None)
    return ModelConfig(
        model_id=model_id,
        quant=(getattr(cfg, "quant", None) or "none").lower(),
        dtype=getattr(cfg, "dtype", "bf16"),
        device=getattr(cfg, "device", None),
        local_files_only=bool(getattr(cfg, "local_files_only", True)),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )

logger = get_logger(__name__)

def build_generators_and_app(models: List[str]):
    generators: List[Any] = []
    enabled_apis: Set[str] = set()
    logger.info("Building generators and Flask app", extra={"stage": "init"})

    # LLM / Chat
    llm_cfg = None
    if "llm-instruct" in models or "llm" in models:
        if "llm-instruct" in models:
            llm_cfg = ModelLLMInstruct()
            enabled_apis.add("chat")
        if "llm" in models:
            # If only llm is present, enable completion;
            # If both present, also expose completion (served by instruct model).
            enabled_apis.add("completion")
            if llm_cfg is None:
                llm_cfg = ModelLLM()
        if llm_cfg is not None:
            model_config = _to_model_config(llm_cfg)
            llm_generator = LLMGenerator(llm_jobs_queue, model_config)
            llm_generator.start()
            generators.append(llm_generator)

    # Images (Stable Diffusion)
    if "images" in models:
        img_cfg = ModelDiffusers()
        img_model_config = _to_model_config(img_cfg)
        img_gen = DiffusersGenerator(image_jobs_queue, img_model_config)
        img_gen.start()
        generators.append(img_gen)
        enabled_apis.add("images")

    # Speech (TTS)
    if "speech" in models:
        tts_cfg = ModelTTS()
        tts_model_config = _to_model_config(tts_cfg)
        sp_gen = SpeechGenerator(speech_jobs_queue, tts_model_config)
        sp_gen.start()
        generators.append(sp_gen)
        enabled_apis.add("speech")

    deps = Deps(
        # queues
        llm_jobs_queue=llm_jobs_queue,
        image_jobs_queue=image_jobs_queue,
        speech_jobs_queue=speech_jobs_queue,
        audio_jobs_queue=audio_jobs_queue,
        # registries
        llm_jobs=llm_jobs,
        image_jobs=image_jobs,
        speech_jobs=speech_jobs,
        audio_jobs=audio_jobs,
        # enabled apis
        enabled_apis=enabled_apis,
    )

    app = create_app(deps)
    return generators, app


def main():
    # Set global logging level from config and propagate
    try:
        set_global_log_level(LOG_LEVEL)
    except Exception:
        # Fallback: default set by logger module
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        choices=["llm", "llm-instruct", "images", "speech"],
        nargs="+",
        default=["llm"],
        help=(
            "One or more models to run: llm (completion), llm-instruct (chat), "
            "images (Stable Diffusion), speech (TTS). Example: --models llm images"
        ),
    )
    args = parser.parse_args()

    logger.info("Starting application", extra={"models": args.models})
    generators, flask_app = build_generators_and_app(args.models)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Running Flask app", extra={"host": host, "port": port})
    try:
        flask_app.run(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
    except Exception:
        logger.exception("Unhandled exception in Flask app")
    finally:
        for g in generators:
            try:
                logger.debug("Stopping generator", extra={"generator": type(g).__name__})
                g.stop()
            except Exception:
                logger.warning("Failed to stop generator", exc_info=True)
            try:
                g.join(timeout=5.0)
            except Exception:
                logger.warning("Failed to join generator", exc_info=True)


if __name__ == "__main__":
    main()


