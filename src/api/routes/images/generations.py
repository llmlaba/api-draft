from flask import Blueprint, jsonify, request
import uuid

from src.generators.job import imageJob
from src.logger import get_logger, correlation_context

logger = get_logger(__name__)


def create_blueprint(deps):
    # Images generation endpoint
    bp = Blueprint("images_generations", __name__, url_prefix="/v1/images/generations")

    job_queue = getattr(deps, "image_jobs_queue", None)
    jobs = getattr(deps, "image_jobs", None)

    @bp.route("/", methods=["POST"])
    def generate_image():
        """
        Request JSON:
        {
          "model": "",
          "prompt": "cat sitting on a chair",
          "size": "512x512",
          "steps": 30,
          "guidance_scale": 7.5,
          "seed": 123
        }
        """
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            logger.info("Image generation request missing prompt")
            return jsonify({"error": "prompt is required"}), 400

        params = {
            "size": data.get("size", "512x512"),
            "steps": int(data.get("steps", 30)),
            "guidance_scale": float(data.get("guidance_scale", 7.5)),
        }
        if data.get("seed") is not None:
            try:
                params["seed"] = int(data.get("seed"))
            except Exception:
                logger.warning("Invalid seed provided; ignoring", exc_info=True)

        if job_queue is None or jobs is None:
            return jsonify({"error": "image generation is not enabled"}), 503

        job_id = str(uuid.uuid4())
        logger.info("Enqueue image generation job", extra={"job_id": job_id})
        logger.debug("Image params", extra={"job_id": job_id, "params": params, "prompt_len": len(prompt)})
        job = imageJob(id=job_id, prompt=prompt, params=params, api="generation")
        jobs[job_id] = job
        job_queue.put(job)

        with correlation_context(job_id):
            job.done.wait()
            if job.error:
                logger.error("Image job failed", extra={"job_id": job_id, "error": job.error})
                return jsonify({"job_id": job_id, "error": job.error}), 500
            logger.info("Image job completed", extra={"job_id": job_id})
            return jsonify({"job_id": job_id, "result": job.result})

    return bp
