from flask import Blueprint, jsonify, request
import uuid

from src.generators.job import llmJob


def create_blueprint(deps):

    bp = Blueprint("completions", __name__, url_prefix="/v1/completion")

    job_queue = deps.llm_jobs_queue
    jobs = deps.llm_jobs

    @bp.route("/", methods=["POST"])
    def generate():
        """
        Тело запроса (JSON):
        {
          "prompt": "...",
          "max_tokens": 256,
          "temperature": 0.7,
          "top_p": 0.95,
        }
        """
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        params = {
            "max_tokens": int(data.get("max_tokens", 256)),
            "temperature": float(data.get("temperature", 0.7)),
            "top_p": float(data.get("top_p", 0.95)),
            "do_sample": bool(data.get("do_sample", True)),
        }

        # Создаём задачу и отправляем в очередь LLM
        job_id = str(uuid.uuid4())
        job = llmJob(id=job_id, prompt=prompt, params=params)
        jobs[job_id] = job
        job_queue.put(job)

        # Синхронный режим: ждём завершения
        job.done.wait()
        if job.error:
            return jsonify({"job_id": job_id, "error": job.error}), 500
        return jsonify({"job_id": job_id, "result": job.result})

    return bp
