# server.py
from flask import Flask
from dataclasses import dataclass
from typing import Optional

from .routes import completions
from .routes.chat import completions as chat_completions
from .routes.images import generations as images_generations
from . import metrics

@dataclass
class Deps:
    llm_jobs_queue: Optional[object] = None
    llm_jobs: Optional[object] = None
    image_jobs_queue: Optional[object] = None
    image_jobs: Optional[object] = None
    api_mode: str = "completion"  # "completion" | "chat" | "images"


def create_app(
    llm_jobs_queue=None,
    llm_jobs=None,
    api_mode: str = "completion",
    image_jobs_queue=None,
    image_jobs=None,
) -> Flask:
    app = Flask(__name__)
    deps = Deps(
        llm_jobs_queue=llm_jobs_queue,
        llm_jobs=llm_jobs,
        image_jobs_queue=image_jobs_queue,
        image_jobs=image_jobs,
        api_mode=api_mode,
    )

    # Базовые метрики/хелсчек
    app.register_blueprint(metrics.create_blueprint(deps))

    # Регистрируем нужный API
    if api_mode == "chat":
        app.register_blueprint(chat_completions.create_blueprint(deps))
    elif api_mode == "images":
        app.register_blueprint(images_generations.create_blueprint(deps))
    else:
        app.register_blueprint(completions.create_blueprint(deps))

    app.extensions["deps"] = deps

    return app

