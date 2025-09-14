# server.py
from flask import Flask
from dataclasses import dataclass

from .routes import completions
from .routes.chat import completions as chat_completions
from . import metrics

@dataclass
class Deps:
    llm_jobs_queue: object
    llm_jobs: object
    api_mode: str = "completion"  # "completion" | "chat"


def create_app(llm_jobs_queue, llm_jobs, api_mode: str = "completion") -> Flask:
    app = Flask(__name__)
    deps = Deps(llm_jobs_queue=llm_jobs_queue, llm_jobs=llm_jobs, api_mode=api_mode)

    # Базовые метрики/хелсчек
    app.register_blueprint(metrics.create_blueprint(deps))

    # Регистрируем нужный API
    if api_mode == "chat":
        app.register_blueprint(chat_completions.create_blueprint(deps))
    else:
        app.register_blueprint(completions.create_blueprint(deps))

    app.extensions["deps"] = deps

    return app

