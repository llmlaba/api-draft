from flask import Blueprint, jsonify, request, send_file
import uuid
import io

from src.generators.job import speechJob


def create_blueprint(deps):
    bp = Blueprint("audio_speech", __name__, url_prefix="/v1/audio/speech")

    job_queue = getattr(deps, "speech_jobs_queue", None)
    jobs = getattr(deps, "speech_jobs", None)

    @bp.route("/", methods=["POST"])
    def synthesize_speech():
        """
        Request JSON example:
        {
          "model": "",
          "input": "Hello! It is example of text to speech conversion.",
          "voice": "v2/en_speaker_6",
          "format": "wav"
        }
        """
        if job_queue is None or jobs is None:
            return jsonify({"error": "speech synthesis is not enabled"}), 503

        data = request.get_json(force=True) or {}
        text = (data.get("input") or data.get("prompt") or "").strip()
        if not text:
            return jsonify({"error": "input is required"}), 400

        params = {
            "input": text,
            "voice": data.get("voice", "v2/en_speaker_6"),
            "format": data.get("format", "wav"),
        }

        job_id = str(uuid.uuid4())
        job = speechJob(id=job_id, prompt=text, params=params)
        jobs[job_id] = job
        job_queue.put(job)

        job.done.wait()
        if job.error:
            return jsonify({"job_id": job_id, "error": job.error}), 500

        result = job.result or {}
        audio_bytes = result.get("audio_bytes")
        fmt = (result.get("format") or "wav").lower()
        mime = result.get("mime") or ("audio/wav" if fmt == "wav" else "application/octet-stream")

        if not audio_bytes:
            return jsonify({"job_id": job_id, "error": "no audio produced"}), 500

        return send_file(
            io.BytesIO(audio_bytes),
            mimetype=mime,
            as_attachment=False,
        )

    return bp
