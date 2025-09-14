# server.py
from flask import Flask
import importlib, pkgutil, os
from dataclasses import dataclass

from . import routes
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

    # Автоподхват всех модулей в /routes/*
    for _, modname, ispkg in pkgutil.iter_modules(routes.__path__):
        if ispkg:
            continue
        module = importlib.import_module(f"{routes.__name__}.{modname}")
        # 1) предпочтительно: фабрика с зависимостями
        if hasattr(module, "create_blueprint"):
            app.register_blueprint(module.create_blueprint(deps))
        # 2) альтернативно: функция register(app)
        elif hasattr(module, "register"):
            module.register(app)

    # Поддержка внешних плагинов через переменную окружения (опционально)
    # Например: EXTRA_ROUTE_PACKAGES="payments.routes,users.routes"
    for pkg in filter(None, map(str.strip, os.getenv("EXTRA_ROUTE_PACKAGES", "").split(","))):
        m = importlib.import_module(pkg)
        if hasattr(m, "bp"):
            app.register_blueprint(m.bp)
        elif hasattr(m, "register"):
            m.register(app)

    app.extensions["deps"] = deps

    return app

