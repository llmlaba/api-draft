"""LLM генератор: класс-обёртка над загрузчиком модели и воркером очереди.

Требования:
- При инициализации принимает очередь задач и конфигурацию ModelConfig.
- Внутри инициализирует модель из src/models/llm.py (llm_loader) и создаёт воркер-поток.
- Поток НЕ запускается автоматически — объект возвращается вызывающему коду.

Использование:
    from queue import Queue
    from src.models.config import ModelConfig
    from src.generators.job import llmJob
    from src.generators.llm import LLMGenerator

    jobs_q: Queue[llmJob] = Queue()
    cfg = ModelConfig(model_id="mistralai/Mistral-7B-Instruct-v0.2", quant="none", dtype="bf16")

    gen = LLMGenerator(jobs_q, cfg)  # воркер создан, но не запущен
    gen.start()                      # явный запуск воркера
    # ... добавляйте задачи в очередь jobs_q ...
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

import torch

from src.models.config import ModelConfig
from src.models.llm import llm_loader
from src.generators.job import llmJob


class LLMGenerator:
    """Класс LLM-генератора с внутренним воркером.

    - job_queue: очередь задач llmJob
    - config: конфигурация модели (ModelConfig)
    """

    def __init__(self, job_queue: "queue.Queue[llmJob]", config: ModelConfig) -> None:
        self.queue: "queue.Queue[llmJob]" = job_queue
        self.config = config

        # Инициализируем модель/токенайзер через унифицированный загрузчик
        loader = llm_loader(config)
        self.model, self.tokenizer = loader.load()

        # Флаги управления потоком и сам поток
        self._stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)

    # Управление жизненным циклом воркера
    def start(self) -> "LLMGenerator":
        if not self.worker.is_alive():
            self.worker.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self.worker.join(timeout)

    # Основной цикл обработки задач
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self._generate(job)
                job.result = result
            except Exception as e:  # noqa: BLE001 — возвращаем ошибку в объект задания
                job.error = repr(e)
            finally:
                job.done.set()
                try:
                    self.queue.task_done()
                except Exception:
                    pass

    # Генерация текста для отдельного задания
    def _generate(self, job: llmJob) -> str:
        prompt = job.prompt
        params = job.params or {}

        max_new_tokens = int(params.get("max_tokens", 256))
        temperature = float(params.get("temperature", 0.7))
        temperature = max(temperature, 1e-8)
        top_p = float(params.get("top_p", 0.95))
        do_sample = bool(params.get("do_sample", True))

        # Подготавливаем входные тензоры и переносим на устройство модели
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

        # Возвращаем только сгенерированную часть (без префикса-подсказки)
        seq = output_ids[0]
        in_len = inputs["input_ids"].shape[1]
        gen_part = seq[in_len:]
        text = self.tokenizer.decode(gen_part, skip_special_tokens=True)
        if not text:  # на всякий случай — вернём полную строку
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
        return text



