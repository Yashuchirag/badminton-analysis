"""
Session-based chunked processing shared by the upload and live modes.

One persistent ScoringEngine per session: the court is calibrated once on the
first chunk and all score/trajectory state carries across chunks. Chunks are
processed in sequence order by a single worker thread. Between chunks the
global frame index advances by CHUNK_FRAME_PAD (one past the engine's
_HIT_GAP_MAX) so cross-boundary velocity math is skipped instead of producing
false hits.
"""
import asyncio
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2

from llm.scoring_engine import ScoringEngine

CHUNK_FRAME_PAD = 9     # one past ScoringEngine._HIT_GAP_MAX
BATCH_WINDOW = 32       # frames per batched GPU detection call (16 at stride 2)
SESSION_TTL = 600       # seconds of inactivity before a session is evicted

SESSIONS: Dict[str, "ChunkSession"] = {}


class ChunkSession:
    def __init__(
        self,
        session_id: str,
        engine_kwargs: dict,
        output_dir: Path,
        annotate: bool = True,
        live: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.session_id = session_id
        self.engine_kwargs = engine_kwargs
        self.output_dir = output_dir
        self.annotate = annotate
        self.live = live
        self.loop = loop

        self.engine: Optional[ScoringEngine] = None
        self.calibrated = False
        self.finished = False           # no more chunks will arrive
        self.global_frame_idx = 0
        self.expected_total_frames = 0  # known upfront for uploads, 0 for live
        self.last_activity = time.time()
        self.landing_events: list = []
        self.output_path = str(output_dir / f"{session_id}_result.mp4") if annotate else None

        self.latest_state: dict = {
            "status": "starting",
            "frame": 0,
            "total_frames": 0,
            "progress_percent": 0.0,
            "score": [0, 0],
            "rally_state": "SERVE_PENDING",
            "last_hitter_side": None,
            "backlog": 0,
        }
        self._subscribers: list = []

        self._pending: Dict[int, str] = {}  # seq -> chunk file path
        self._next_seq = 0
        self._cond = threading.Condition()

        self._writer: Optional[cv2.VideoWriter] = None
        self._write_q: Optional[queue.Queue] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._fps = 30.0

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── Public API (called from the event loop) ──────────────────────────────

    def add_chunk(self, seq: int, path: str) -> int:
        if self.finished:
            raise RuntimeError("Session already finished")
        with self._cond:
            self._pending[seq] = path
            backlog = len(self._pending)
            self._cond.notify()
        self.last_activity = time.time()
        self.latest_state["backlog"] = backlog
        return backlog

    def finish(self) -> None:
        with self._cond:
            self.finished = True
            self._cond.notify()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._set_state(status="loading")
            self.engine = ScoringEngine(**self.engine_kwargs)
            self._set_state(status="calibrating")
            self._broadcast()
            while True:
                path = self._next_chunk()
                if path is None:
                    break
                self._process_chunk(path)
                self.last_activity = time.time()
            self._finalize()
        except Exception as exc:
            print(f"[session error] {self.session_id}: {exc}")
            self._close_writer()
            self._cleanup_chunks()
            self._set_state(status="error", message=str(exc))
            self._broadcast()
            self.engine = None

    def _next_chunk(self) -> Optional[str]:
        with self._cond:
            while True:
                if self._next_seq in self._pending:
                    path = self._pending.pop(self._next_seq)
                    self._next_seq += 1
                    return path
                if self.finished:
                    if not self._pending:
                        return None
                    # a chunk was never uploaded; skip the gap
                    self._next_seq = min(self._pending)
                    continue
                self._cond.wait(timeout=1.0)

    def _process_chunk(self, path: str) -> None:
        if not self.calibrated:
            try:
                self.engine.auto_calibrate(path, n_frames=60)
                self.calibrated = True
            except Exception as exc:
                if not self.live:
                    raise
                # Live mode: discard this chunk and retry calibration on the
                # next one while the user adjusts framing.
                os.remove(path)
                self._set_state(status="calibrating", message=str(exc))
                self._broadcast()
                return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open chunk {path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._set_state(status="processing")

        while True:
            window = []
            while len(window) < BATCH_WINDOW:
                ret, frame = cap.read()
                if not ret:
                    break
                window.append(frame)
            if not window:
                break
            dets = self.engine.precompute_detections(window, self.global_frame_idx)
            for frame in window:
                event = self.engine.process_frame(
                    frame, self.global_frame_idx,
                    det=dets.get(self.global_frame_idx),
                )
                if event is not None:
                    self.landing_events.append({
                        "frame": event.frame_idx,
                        "verdict": event.verdict.value,
                        "court_x": round(event.court_x, 3),
                        "court_y": round(event.court_y, 3),
                        "confidence": round(event.confidence, 3),
                        "source": event.source,
                        "reason": event.reason,
                    })
                if self.annotate:
                    self._enqueue_annotated(frame)
                self.global_frame_idx += 1
                self.latest_state["frame"] += 1
                self.latest_state["score"] = list(self.engine.state.score)
                self.latest_state["rally_state"] = self.engine.state.rally_state
                self.latest_state["last_hitter_side"] = self.engine.state.last_hitter_side
                if self.expected_total_frames:
                    self.latest_state["progress_percent"] = round(
                        self.latest_state["frame"] / self.expected_total_frames * 100, 1
                    )
            if len(window) < BATCH_WINDOW:
                break

        cap.release()
        os.remove(path)
        self.global_frame_idx += CHUNK_FRAME_PAD
        self._set_state(status="processing", backlog=len(self._pending))
        self._broadcast()

    def _finalize(self) -> None:
        self._close_writer()
        self._cleanup_chunks()
        self._set_state(
            status="complete",
            progress_percent=100.0,
            game=self.engine.state.game if self.engine else 1,
            landing_events=self.landing_events,
            total_landings=len(self.landing_events),
            output_video=self.output_path if self.annotate else None,
        )
        self._broadcast()
        self.engine = None  # release model VRAM; results live on in latest_state

    def _cleanup_chunks(self) -> None:
        with self._cond:
            leftover = list(self._pending.values())
            self._pending.clear()
        for p in leftover:
            try:
                os.remove(p)
            except OSError:
                pass
        if self.live:
            try:
                (self.output_dir / self.session_id).rmdir()
            except OSError:
                pass

    # ── Annotated output (writer thread keeps cv2 writes off inference) ──────

    def _enqueue_annotated(self, frame) -> None:
        frame = self.engine.draw_court_boundaries(frame)
        if self.engine.last_shuttle is not None:
            x, y = int(self.engine.last_shuttle[0]), int(self.engine.last_shuttle[1])
            cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
        s = self.engine.state.score
        cv2.putText(frame, f"{s[0]} - {s[1]}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError("Cannot create output video writer")
            self._write_q = queue.Queue(maxsize=32)
            self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self._writer_thread.start()
        self._write_q.put(frame)

    def _write_loop(self) -> None:
        while True:
            frame = self._write_q.get()
            if frame is None:
                break
            self._writer.write(frame)

    def _close_writer(self) -> None:
        if self._writer_thread is not None:
            self._write_q.put(None)
            self._writer_thread.join()
            self._writer_thread = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ── State + broadcast ─────────────────────────────────────────────────────

    def _set_state(self, **kw) -> None:
        if self.engine is not None:
            self.latest_state["score"] = list(self.engine.state.score)
            self.latest_state["rally_state"] = self.engine.state.rally_state
            self.latest_state["last_hitter_side"] = self.engine.state.last_hitter_side
        self.latest_state["total_frames"] = self.expected_total_frames
        self.latest_state.update(kw)

    def _broadcast(self) -> None:
        if self.loop is None or not self._subscribers:
            return
        state = dict(self.latest_state)
        for q in list(self._subscribers):
            self.loop.call_soon_threadsafe(q.put_nowait, state)


# ── Module-level session registry ────────────────────────────────────────────

def create_session(session_id: str, **kwargs) -> ChunkSession:
    cleanup_idle()
    session = ChunkSession(session_id, **kwargs)
    SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> Optional[ChunkSession]:
    return SESSIONS.get(session_id)


def active_live_sessions() -> int:
    return sum(
        1 for s in SESSIONS.values()
        if s.live and s.latest_state["status"] not in ("complete", "error")
    )


def active_sessions() -> int:
    """Sessions still holding an engine (any mode). VRAM allows one at a time.

    A session is only inactive once BOTH signals agree: the worker has
    released the engine AND set a terminal status. Checking the engine alone
    would miss sessions still loading models (engine not yet assigned), and
    checking the status alone can lag behind the engine release.
    """
    return sum(
        1 for s in SESSIONS.values()
        if s.engine is not None
        or s.latest_state["status"] not in ("complete", "error")
    )


def cleanup_idle(ttl: int = SESSION_TTL) -> None:
    now = time.time()
    for sid, s in list(SESSIONS.items()):
        if now - s.last_activity > ttl:
            s.finish()
            del SESSIONS[sid]
