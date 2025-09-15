# server.py
from flask import Flask
from dataclasses import dataclass, field
from typing import Optional, Set

from .routes import completions
from .routes.chat import completions as chat_completions
from .routes.images import generations as images_generations
from .routes.audio import speech as audio_speech
from . import metrics

@dataclass
class Deps:
    # Queues
    llm_jobs_queue: Optional[object] = None
    image_jobs_queue: Optional[object] = None
    speech_jobs_queue: Optional[object] = None
    audio_jobs_queue: Optional[object] = None  # reserved for future (ASR)

    # Job registries
    llm_jobs: Optional[object] = None
    image_jobs: Optional[object] = None
    speech_jobs: Optional[object] = None
    audio_jobs: Optional[object] = None  # reserved for future (ASR)

    # Which APIs to expose
    enabled_apis: Set[str] = field(default_factory=set)  # {"completion", "chat", "images", "speech"}


def create_app(deps: Deps) -> Flask:
    app = Flask(__name__)

    # Base metrics/healthcheck
    app.register_blueprint(metrics.create_blueprint(deps))

    # Conditionally register APIs based on enabled_apis
    if "chat" in deps.enabled_apis:
        app.register_blueprint(chat_completions.create_blueprint(deps))
    if "completion" in deps.enabled_apis:
        app.register_blueprint(completions.create_blueprint(deps))
    if "images" in deps.enabled_apis:
        app.register_blueprint(images_generations.create_blueprint(deps))
    if "speech" in deps.enabled_apis:
        app.register_blueprint(audio_speech.create_blueprint(deps))

    app.extensions["deps"] = deps
    return app

