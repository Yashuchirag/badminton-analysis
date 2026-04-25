from __future__ import annotations

import base64
import json
from typing import Optional

import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠  ollama package not found. Run: pip install ollama")

JUDGE_SYSTEM_PROMPT = """\
You are a hawk-eye line judge for badminton. You receive the shuttle's court coordinates and distance to the boundary line.
Respond ONLY with a JSON object, no markdown, no explanation:
{"verdict": "IN" or "OUT", "confidence": 0.0-1.0, "reason": "one short sentence"}
"""


class LineJudge:
    """
    Sends shuttle landing coordinates + annotated frame to Gemma and returns
    an IN/OUT verdict. Always blocking — only called for near-line shots so
    latency is acceptable.
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
        Falls back to {"verdict": "OUT", ...} on any error.
        """
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode("utf-8")

        user_msg = (
            f"Zone: {zone_name}\n"
            f"Shuttle court position: x={court_x:.3f}m, y={court_y:.3f}m\n"
            f"Distance to boundary: {distance_to_line * 100:.1f}cm "
            f"({'inside' if distance_to_line > 0 else 'outside'})\n"
            "The cyan polygon drawn on the image shows the court boundary. "
            "The red crosshair marks the shuttle landing position.\n"
            "Is the shuttle IN or OUT? The shuttle is ON the line if distance is within 2cm."
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg, "images": [b64]},
        ]

        try:
            response = ollama.chat(model=self.model, messages=messages, stream=False)
            raw = response["message"]["content"].strip()

            # Strip markdown fences if Gemma wraps output in ```json ... ```
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
