import os
import uuid

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/v1/completion", methods=["POST"])
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
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    params = {
        "max_tokens": int(data.get("max_tokens", 256)),
        "temperature": float(data.get("temperature", 0.7)),
        "top_p": float(data.get("top_p", 0.95)),
        "do_sample": bool(data.get("do_sample", True)),
    }

    # Создаём задачу
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, prompt=prompt, params=params)
    jobs[job_id] = job
    job_queue.put(job)

    # Блокируемся до готовности (или можно добавить timeout)
    job.done.wait()
    if job.error:
        return jsonify({"job_id": job_id, "error": job.error}), 500
    return jsonify({"job_id": job_id, "result": job.result})

    # Асинхронный режим: клиент заберёт результат по /result/<job_id>
    return jsonify({"job_id": job_id, "status": "queued"}), 202

if __name__ == "__main__":
    # Для разработки: flask dev-сервер
    # В проде используйте gunicorn:  gunicorn -w 1 -k gthread --threads 16 app:app
    # ВАЖНО: workers=1 (один процесс), иначе каждая копия загрузит модель в VRAM.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), threaded=True)
