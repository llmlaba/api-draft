import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

generator = None  # подмените вашим реальным генератором

def init_text_model():
    global text_pipe
    if text_pipe is None:
        # Mistral 7B Instruct как пример
        text_pipe = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            device=DEVICE
        )
        text_pipe.model.eval()

# === 2) СТРУКТУРЫ ДАННЫХ ДЛЯ ОЧЕРЕДИ ===

@dataclass
class Job:
    id: str
    prompt: str
    params: Dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    error: Optional[str] = None
    created_ts: float = field(default_factory=time.time)

# Очередь задач и реестр
job_queue: "queue.Queue[Job]" = queue.Queue()
jobs: Dict[str, Job] = {}

# === 3) ВОРКЕР ===

def worker_loop():
    init_model_once()
    while True:
        job: Job = job_queue.get()
        if job is None:  # сигнал на завершение (необязательно)
            break
        try:
            # ВАЖНО: доступ к модели только из этого потока
            # Можно обернуть в torch.inference_mode() при PyTorch
            # with torch.inference_mode():
            out = generator(
                job.prompt,
                max_new_tokens=job.params.get("max_tokens", 256),
                temperature=max(job.params.get("temperature", 0.7), 1e-8),
                top_p=job.params.get("top_p", 0.95),
                do_sample=job.params.get("do_sample", True),
                return_full_text=False
            )[0]["generated_text"]
            job.result = out
        except Exception as e:
            job.error = repr(e)
        finally:
            job.done.set()
            job_queue.task_done()

worker = threading.Thread(target=worker_loop, daemon=True)
worker.start()
