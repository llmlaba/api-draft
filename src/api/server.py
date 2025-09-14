# server.py
from flask import Flask
import importlib, pkgutil, os
from dataclasses import dataclass

from .routes import completions
from . import metrics

@dataclass
class Deps:
    llm_jobs_queue: object
    llm_jobs: object


def create_app(llm_jobs_queue, llm_jobs) -> Flask:
    app = Flask(__name__)
    deps = Deps(llm_jobs_queue=llm_jobs_queue, llm_jobs=llm_jobs)

    # Базовые метрики/хелсчек
    app.register_blueprint(metrics.create_blueprint(deps))
    # Роуты для работы с LLM
    app.register_blueprint(completions.create_blueprint(deps))

    app.extensions["deps"] = deps

    return app

