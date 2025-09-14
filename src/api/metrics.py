# metrics.py
from flask import Blueprint, jsonify

def create_blueprint(deps):
    bp = Blueprint("metrics", __name__)

    job_queue = deps.llm_jobs_queue

    @bp.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "queue_size": job_queue.qsize()})
