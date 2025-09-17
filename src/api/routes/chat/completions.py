from flask import Blueprint, jsonify, request
import uuid

from src.generators.job import llmJob
from src.logger import get_logger, correlation_context

logger = get_logger(__name__)


def create_blueprint(deps):
    # Chat Completions endpoint (plural)
    bp = Blueprint("chat_completions", __name__, url_prefix="/v1/chat/completions")

    job_queue = deps.llm_jobs_queue
    jobs = deps.llm_jobs

    @bp.route("/", methods=["POST"])
    def generate_chat():
        """
        JSON body example:
        {
          "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Hello"}
          ],
          "max_tokens": 256,
          "temperature": 0.7,
          "top_p": 0.95,
          "do_sample": true
        }
        """
        data = request.get_json(force=True) or {}
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.info("Chat completion request missing messages list")
            return jsonify({"error": "messages (non-empty list) is required"}), 400

        params = {
            "max_tokens": int(data.get("max_tokens", 256)),
            "temperature": float(data.get("temperature", 0.7)),
            "top_p": float(data.get("top_p", 0.95)),
            "do_sample": bool(data.get("do_sample", True)),
        }

        job_id = str(uuid.uuid4())
        logger.info("Enqueue chat-completion job", extra={"job_id": job_id})
        logger.debug("Chat params", extra={"job_id": job_id, "params": params, "num_messages": len(messages)})
        job = llmJob(
            id=job_id,
            api="chat-completion",
            prompt="",
            messages=messages,
            params=params,
        )
        jobs[job_id] = job
        job_queue.put(job)

        with correlation_context(job_id):
            job.done.wait()
            if job.error:
                logger.error("Chat job failed", extra={"job_id": job_id, "error": job.error})
                return jsonify({"job_id": job_id, "error": job.error}), 500
            logger.info("Chat job completed", extra={"job_id": job_id})
            return jsonify({"job_id": job_id, "result": job.result})

    return bp

