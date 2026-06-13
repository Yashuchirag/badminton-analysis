# pyrefly: ignore [missing-import]
"""
Integrated test: shuttle tracking (model2/train_and_track.py style trail)
+ court tracking overlay + geometric scoring, in a single inference pass.

Run from backend/:  ./ubenv/bin/python llm/integrated_main.py
Optional frame cap for quick tests:  MAX_FRAMES=300 ./ubenv/bin/python llm/integrated_main.py
"""
import cv2
import sys, os, time
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring_engine import ScoringEngine

# ── Video to process ──────────────────────────────────────────────────────────
VIDEO_PATH  = "model2/dataset/TrackNetV2_Dataset/Professional/match5/video/merged_20260520_174812.mp4"
OUTPUT_PATH = "integrated_output.mp4"
MAX_FRAMES  = int(os.environ.get("MAX_FRAMES", 0))   # 0 = full video

TRAIL_LENGTH = 30
BANNER_HOLD  = 60   # frames to keep the landing verdict banner on screen

engine = ScoringEngine(
    obb_weights="model2/models/claudette_trained/yolo_obb_finetune/weights/best.pt",
    tracknet_weights="model2/models/claudette_trained/tracknetv2/tracknetv2_best.pth",
    mode="singles",
    tracking_mode="hybrid",
    device="0",
)

print("Auto-detecting court lines…")
try:
    engine.auto_calibrate(VIDEO_PATH)
except RuntimeError as e:
    print(f"\n❌ Court auto-calibration failed:\n  {e}")
    print("   Check that the video shows clear court lines.")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)

fps          = int(cap.get(cv2.CAP_PROP_FPS))
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
out          = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

trail = deque(maxlen=TRAIL_LENGTH)
banner = None          # (LandingEvent, frames_remaining)
frame_idx = 0
t_start = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MAX_FRAMES and frame_idx >= MAX_FRAMES:
            break

        if frame_idx > 0 and frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            print(f"[{frame_idx}/{total_frames}] {frame_idx / elapsed:.1f} fps | "
                  f"score={engine.state.score} | {engine.state.rally_state}")

        landing = engine.process_frame(frame, frame_idx)
        score   = engine.state.score
        if landing:
            banner = (landing, BANNER_HOLD)
            print(f"[{frame_idx}] {landing.verdict.value} | score={score} | "
                  f"src={landing.source} | {landing.reason}")

        # ── Court tracking overlay ────────────────────────────────────────
        frame = engine.draw_court_boundaries(frame)

        # ── Shuttle tracking: marker + fading trail (track_video style) ───
        if engine.last_shuttle is not None:
            sx, sy, sconf, ssrc = engine.last_shuttle
            x, y = int(sx), int(sy)
            trail.append((x, y))
            cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
            cv2.putText(frame, f"{ssrc} {sconf:.2f}", (x + 16, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif engine.state.rally_state == "SERVE_PENDING":
            trail.clear()   # rally over (or cooldown) — drop the stale trail

        if len(trail) > 1:
            pts = list(trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (0, int(255 * alpha), int(100 * (1 - alpha)))
                # cv2.line(frame, pts[i - 1], pts[i], color, 2)

        # ── Scoreboard ────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (width, 70), (0, 0, 0), -1)
        cv2.putText(frame, "P1", (20, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, str(score[0]), (75, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 100), 2)
        cv2.putText(frame, "-", (120, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 200), 2)
        cv2.putText(frame, str(score[1]), (145, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 100), 2)
        cv2.putText(frame, "P2", (195, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Game {engine.state.game}", (width - 160, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 200, 200), 2)

        serving = engine.state.serving_side
        serve_x = 85 if serving == 0 else 150
        cv2.circle(frame, (serve_x, 58), 5, (0, 255, 255), -1)

        # ── Rally state + last hitter ─────────────────────────────────────
        rs_color = (0, 200, 0) if engine.state.rally_state == "RALLY_ACTIVE" else (0, 140, 255)
        lh = engine.state.last_hitter_side or "?"
        cv2.putText(frame, f"{engine.state.rally_state} | hitter={lh} | frame {frame_idx}",
                    (10, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rs_color, 1)

        # ── Landing verdict banner (held for BANNER_HOLD frames) ──────────
        if banner is not None:
            ev, remaining = banner
            color = (0, 220, 0) if ev.verdict.value == "IN" else (0, 0, 220)
            cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), -1)
            cv2.putText(frame, f"  {ev.verdict.value}  {ev.reason[:60]}",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.circle(frame, (int(ev.pixel_x), int(ev.pixel_y)), 10, color, 2)
            banner = (ev, remaining - 1) if remaining > 1 else None

        out.write(frame)
        frame_idx += 1

except KeyboardInterrupt:
    print(f"\nStopped at frame {frame_idx} | Score: {engine.state.score}")

finally:
    cap.release()
    out.release()
    print(f"\n✓ Saved → {OUTPUT_PATH}")
    print(f"  Final score : {engine.state.score}")
    print(f"  Total points: {len(engine.state.history)}")
