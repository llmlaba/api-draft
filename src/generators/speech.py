"""Speech (TTS) generator: queue-based worker using Bark via tts_loader.

- Accepts speechJob tasks from a dedicated Queue.
- Produces base64-encoded audio (wav by default) with simple metadata.
"""
from __future__ import annotations

import base64
import io
import queue
import threading
from typing import Optional, Dict, Any

import torch

from src.models.config import ModelConfig
from src.models.tts import tts_loader
from src.generators.job import speechJob


class SpeechGenerator:
    def __init__(self, job_queue: "queue.Queue[speechJob]", config: ModelConfig) -> None:
        self.queue: "queue.Queue[speechJob]" = job_queue
        self.config = config

        loader = tts_loader(config)
        self.model, self.processor = loader.load()

        self._stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> "SpeechGenerator":
        if not self.worker.is_alive():
            self.worker.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self.worker.join(timeout)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self._synthesize(job)
                job.result = result
            except Exception as e:  # noqa: BLE001
                job.error = repr(e)
            finally:
                job.done.set()
                try:
                    self.queue.task_done()
                except Exception:
                    pass

    def _synthesize(self, job: speechJob) -> Dict[str, Any]:
        params = job.params or {}
        text = job.prompt or params.get("input", "")
        if not text:
            raise ValueError("input text is required for speech synthesis")

        voice = params.get("voice", "v2/en_speaker_6")
        fmt = (params.get("format", "wav") or "wav").lower()
        sample_rate = int(params.get("sample_rate", 22050))

        # Prepare inputs for Bark
        inputs = self.processor(
            text=[text], voice_preset=voice, return_tensors="pt"
        )
        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Generate audio (float32 range -1..1)
        with torch.no_grad():
            audio = self.model.generate(**inputs)

        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()

        # Ensure mono and expected dtype
        import numpy as np

        audio = np.asarray(audio)
        if audio.ndim > 1:
            # Collapse extra dims to mono
            audio = audio.reshape(-1)
        # Normalize/clip to [-1,1]
        audio = np.clip(audio, -1.0, 1.0)

        if fmt == "wav":
            b64 = self._encode_wav_base64(audio, sample_rate)
        else:
            # For unsupported formats, still return WAV
            fmt = "wav"
            b64 = self._encode_wav_base64(audio, sample_rate)

        return {
            "audio": b64,
            "format": fmt,
            "sample_rate": sample_rate,
            "num_samples": int(audio.shape[0]),
        }

    @staticmethod
    def _encode_wav_base64(audio_f32, sample_rate: int) -> str:
        import numpy as np
        import wave

        # Convert float32 -1..1 to int16 PCM
        pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype("<i2")
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return base64.b64encode(bio.getvalue()).decode("ascii")
