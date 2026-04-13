from __future__ import annotations

import base64
import threading
import queue
import json
import datetime
from typing import Optional

import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠  ollama package not found. Run: pip install ollama")

from rally_buffer import RallyBuffer
from event_detector import Event

commentary_file = open("commentary.txt", "a")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a hawk-eye line judge for badminton. You receive the shuttle's court coordinates and distance to the boundary line.
Respond ONLY with a JSON object, no markdown, no explanation:
{"verdict": "IN" or "OUT", "confidence": 0.0-1.0, "reason": "one short sentence"}
"""

COMMENTARY_SYSTEM_PROMPT = """\
You are a live badminton commentator receiving real-time data from a computer vision system.

Rules:
- Maximum 2 sentences, 35 words total. Be punchy and energetic.
- Use present tense — this is LIVE.
- Mention speed when above 100 km/hr. Mention zone (near/far, left/right) when relevant.
- On point_end: briefly summarise the rally (shots, how it ended).
- Do NOT invent details not in the data. Do NOT repeat the previous comment verbatim.
- Sound like a sports broadcaster, not a robot reading a log file.
- Dont hallucinate shuttle speed. Speed of shuttle can only reach upto 500km/hr max.
"""


# ---------------------------------------------------------------------------
# LineJudge — blocking, called synchronously from ScoringEngine
# ---------------------------------------------------------------------------

class LineJudge:
    """
    Sends shuttle landing coordinates + frame to Gemma and returns IN/OUT verdict.
    Always blocking — only called for near-line shots so latency is acceptable.
    """

    def __init__(self, model: str = "gemma4:e4b"):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package required. Run: pip install ollama")
        self.model = model
        print(f"✓ LineJudge ready  model={model}")

    def judge(self, frame: np.ndarray, court_x: float, court_y: float,
              zone_name: str, distance_to_line: float) -> dict:
        """
        Returns {"verdict": "IN"|"OUT", "confidence": float, "reason": str}
        Falls back to OUT on any error.
        """
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode("utf-8")

        user_msg = (
            f"Zone: {zone_name}\n"
            f"Shuttle court position: x={court_x:.3f}m, y={court_y:.3f}m\n"
            f"Distance to boundary: {distance_to_line * 100:.1f}cm "
            f"({'inside' if distance_to_line > 0 else 'outside'})\n"
            "Is the shuttle IN or OUT? The shuttle is ON the line if distance is within 2cm."
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg, "images": [b64]},
        ]

        try:
            response = ollama.chat(model=self.model, messages=messages, stream=False)
            raw = response["message"]["content"].strip()

            # Strip markdown fences if Gemma wraps in ```json
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            result.setdefault("verdict", "OUT")
            result.setdefault("confidence", 0.5)
            result.setdefault("reason", "")
            return result

        except Exception as e:
            print(f"⚠  LineJudge error: {e}")
            return {"verdict": "OUT", "confidence": 0.3, "reason": f"Gemma error: {e}"}


# ---------------------------------------------------------------------------
# Commentator — async, fire-and-forget, drop-oldest queue
# ---------------------------------------------------------------------------

class Commentator:
    """
    Non-blocking commentary. Jobs are queued and processed by a background thread.
    Stale commentary is dropped rather than delivered late.
    """

    def __init__(self, model: str = "gemma4:e4b", max_queue: int = 2,
                 print_commentary: bool = True, callback: Optional[callable] = None):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package required. Run: pip install ollama")

        self.model = model
        self.print_commentary = print_commentary
        self.callback = callback

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._last_comment: str = ""
        self._lock = threading.Lock()
        self._running = True

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print(f"✓ Commentator ready  model={model}")

    def trigger(self, buffer: RallyBuffer, event: Event,
                frame: Optional[np.ndarray] = None):
        """
        Non-blocking. Encodes frame if provided and enqueues the job.
        Drops the oldest job if the queue is full.
        """
        frame_b64 = None
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buf).decode("utf-8")

        with self._lock:
            prev = self._last_comment

        job = {
            "context":      buffer.format_context(),
            "trigger_kind": event.kind,
            "trigger_ts":   event.timestamp,
            "prev_comment": prev,
            "frame_b64":    frame_b64,
        }
        self._enqueue(job)

    def shutdown(self):
        self._running = False
        self._queue.put(None)  # wake worker so it sees the sentinel
        self._worker.join(timeout=5)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enqueue(self, job: dict):
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                pass

    def _worker_loop(self):
        while self._running:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:  # shutdown sentinel
                break
            self._run_job(job)

    def _run_job(self, job: dict):
        user_msg = (
            f"Rally data:\n{job['context']}\n\n"
            f"Trigger: {job['trigger_kind']} at t={job['trigger_ts']:.2f}s\n"
            f"Previous comment (do not repeat): \"{job['prev_comment']}\"\n\n"
            "Give a single live commentary line for this moment."
        )
        messages = [
            {"role": "system", "content": COMMENTARY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        if job["frame_b64"]:
            messages[-1]["images"] = [job["frame_b64"]]

        try:
            full_text = ""
            for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
                full_text += chunk["message"]["content"]

            full_text = full_text.strip()

            if self.print_commentary:
                print(f"\n🎙  [{job['trigger_ts']:.1f}s | {job['trigger_kind']}] {full_text}")

            commentary_file.write(
                f"{datetime.datetime.now()} | {job['trigger_kind']} | {full_text}\n"
            )
            commentary_file.flush()

            with self._lock:
                self._last_comment = full_text

            if self.callback and full_text:
                self.callback(full_text)

        except Exception as e:
            print(f"⚠  Commentator error: {e}")