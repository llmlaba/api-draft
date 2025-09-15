from flask import Blueprint, jsonify

def create_blueprint(deps):
    bp = Blueprint("metrics", __name__)

    # Try to collect sizes of all known queues if present
    def _qsize(q):
        try:
            return int(q.qsize())
        except Exception:
            return 0

    @bp.get("/healthz")
    def healthz():
        breakdown = {}
        total = 0
        for name in (
            "llm_jobs_queue",
            "image_jobs_queue",
            "speech_jobs_queue",
            "audio_jobs_queue",
        ):
            q = getattr(deps, name, None)
            if q is not None:
                size = _qsize(q)
                breakdown[name] = size
                total += size
        return jsonify({"status": "ok", "total_queue_size": total, "queues": breakdown})

    return bp
