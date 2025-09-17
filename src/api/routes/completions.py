from flask import Blueprint, jsonify, request
import uuid

from src.generators.job import llmJob
from src.logger import get_logger, correlation_context

logger = get_logger(__name__)


def create_blueprint(deps):

    bp = Blueprint("completions", __name__, url_prefix="/v1/completions")

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
            logger.info("Completion request missing prompt")
            return jsonify({"error": "prompt is required"}), 400

        params = {
            "max_tokens": int(data.get("max_tokens", 256)),
            "temperature": float(data.get("temperature", 0.7)),
            "top_p": float(data.get("top_p", 0.95)),
            "do_sample": bool(data.get("do_sample", True)),
        }

        # Создаём задачу и отправляем в очередь LLM
        job_id = str(uuid.uuid4())
        logger.info("Enqueue LLM completion job", extra={"job_id": job_id})
        logger.debug("Completion params", extra={"job_id": job_id, "params": params, "prompt_len": len(prompt)})
        job = llmJob(id=job_id, prompt=prompt, params=params)
        jobs[job_id] = job
        job_queue.put(job)

        # Синхронный режим: ждём завершения
        with correlation_context(job_id):
            job.done.wait()
            if job.error:
                logger.error("LLM job failed", extra={"job_id": job_id, "error": job.error})
                return jsonify({"job_id": job_id, "error": job.error}), 500
            logger.info("LLM job completed", extra={"job_id": job_id})
            return jsonify({"job_id": job_id, "result": job.result})

    return bp
