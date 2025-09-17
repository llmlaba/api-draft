"""Diffusers (Stable Diffusion) generator with internal worker thread.

- Accepts a Queue[imageJob]
- Loads pipeline via src/models/diffusers.diffusion_loader
- Generates image for incoming jobs and returns result dict with base64-encoded PNG

Parameters expected in imageJob.params:
- size: string like "512x512" (default "512x512")
- steps: int, num_inference_steps (default 30)
- guidance_scale: float (default 7.5)
- seed: Optional[int]
"""
from __future__ import annotations

import base64
import io
import queue
import threading
from typing import Optional, Tuple

import torch

from src.generators.job import imageJob
from src.models.config import ModelConfig
from src.models.diffusers import diffusion_loader
from src.logger import get_logger, correlation_context


def _parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w_s, h_s = (size_str or "512x512").lower().split("x", 1)
        w = int(w_s)
        h = int(h_s)
        # Clamp to reasonable SD1.5 limits
        w = max(64, min(1536, w))
        h = max(64, min(1536, h))
        # Snap to multiples of 8 for UNet
        w = (w // 8) * 8
        h = (h // 8) * 8
        return w, h
    except Exception:
        return 512, 512


class DiffusersGenerator:
    def __init__(self, job_queue: "queue.Queue[imageJob]", config: ModelConfig) -> None:
        self.queue: "queue.Queue[imageJob]" = job_queue
        self.config = config
        self._log = get_logger(__name__)

        loader = diffusion_loader(config)
        self.pipe = loader.load()

        self._stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> "DiffusersGenerator":
        if not self.worker.is_alive():
            self._log.info("Starting Diffusers worker thread")
            self.worker.start()
        return self

    def stop(self) -> None:
        self._log.info("Stopping Diffusers worker thread")
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self.worker.join(timeout)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with correlation_context(job.id):
                try:
                    self._log.info("Processing image job", extra={"job_id": job.id})
                    self._log.debug("Image job params", extra={"job_id": job.id, "params": job.params})
                    result = self._generate(job)
                    job.result = result
                    self._log.info("Image job done", extra={"job_id": job.id})
                except Exception as e:  # noqa: BLE001
                    job.error = repr(e)
                    self._log.exception("Error while processing image job", extra={"job_id": job.id})
                finally:
                    job.done.set()
                    try:
                        self.queue.task_done()
                    except Exception:
                        self._log.warning("queue.task_done() failed", exc_info=True)

    def _generate(self, job: imageJob):
        params = job.params or {}
        size = params.get("size", "512x512")
        width, height = _parse_size(size)
        steps = int(params.get("steps", 30))
        guidance_scale = float(params.get("guidance_scale", 7.5))
        seed = params.get("seed")
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=str(getattr(self.pipe, "device", "cpu"))).manual_seed(int(seed))
            except Exception:
                generator = torch.Generator().manual_seed(int(seed))

        prompt = job.prompt or params.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required for image generation")

        out = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = out.images[0]

        # Encode to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "data": [
                {
                    "b64_json": b64,
                    "mime": "image/png",
                    "width": width,
                    "height": height,
                }
            ]
        }
