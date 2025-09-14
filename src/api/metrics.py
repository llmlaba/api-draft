@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "queue_size": job_queue.qsize()})