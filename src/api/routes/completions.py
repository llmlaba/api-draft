from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, time, uuid

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

MODEL_PATH = "/llm/mistral"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
).to("cuda")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # GPU
)

print("Model loaded.")

print("Test llm request", generator("What you know about sun?", max_new_tokens=20)[0]["generated_text"])

app = Flask(__name__)

@app.get("/health")
def health():
    return Response("ok", mimetype="text/plain")

def _truncate_at_stop(text: str, stops: list[str]):
    """
    Cut `text` at the earliest occurrence of any stop sequence.

    Args:
        text: generated text to post-process.
        stops: list of stop strings; empty/None items are ignored.

    Returns:
        (truncated_text, finish_reason):
            - truncated_text: text up to the earliest stop (or original text if none found)
            - finish_reason: "stop" if truncated, otherwise None
    """
    if not stops:
        return text, None
    cut_idx = None
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and (cut_idx is None or i < cut_idx):
            cut_idx = i
    if cut_idx is not None:
        return text[:cut_idx], "stop"
    return text, None

def _tok_count(s: str) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))

@app.route("/v1/completion", methods=["POST"])
def completion():
    """
    JSON:
      {
        "prompt": "string",              # required
        "max_tokens": 128,               # optional
        "temperature": 0.7,              # optional
        "top_p": 0.95,                   # optional
        "stop": "\n\n" or ["###"]        # optional
      }
    """
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        return jsonify({"error": {"message": "Field 'prompt' (string) is required"}}), 400

    max_tokens  = int(data.get("max_tokens", 128))
    temperature = float(data.get("temperature", 0.7))
    top_p       = float(data.get("top_p", 0.95))
    stop        = data.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in (stop or []) if isinstance(s, str)]

    do_sample = temperature > 0.0

    compl_id = f"cmpl-{uuid.uuid4().hex}"
    t0 = time.time()

    out = generator(
        prompt,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        do_sample=do_sample,
        return_full_text=False
    )[0]["generated_text"]

    app.logger.info(f"[{compl_id}] {time.time()-t0:.2f}s for {max_tokens} tokens")

    text, finish_reason = _truncate_at_stop(out.lstrip(), stops)
    if finish_reason is None:
        finish_reason = "length"  # простая эвристика

    usage = {
        "prompt_tokens": _tok_count(prompt),
        "completion_tokens": _tok_count(text),
        "total_tokens": _tok_count(prompt) + _tok_count(text),
    }

    resp = {
        "id": compl_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "mistral-7b-local",
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        }],
        "usage": usage
    }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)