# pyrefly: ignore [missing-import]
import cv2
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring_engine import ScoringEngine

# ── Video to process ──────────────────────────────────────────────────────────
VIDEO_PATH = "model2/dataset/TrackNetV2_Dataset/Professional/match3/video/merged_20260520_174744.mp4"
OUTPUT_PATH = "scored_output_3.mp4"

engine = ScoringEngine(
    obb_weights="model2/models/claudette_trained/yolo_obb_finetune/weights/best.pt",
    tracknet_weights="model2/models/claudette_trained/tracknetv2/tracknetv2_best.pth",
    mode="singles",
    tracking_mode="hybrid",
    device="0",
)

# ── Auto-detect court corners from the video — no manual calibration needed ──
print("Auto-detecting court lines…")
try:
    engine.auto_calibrate(VIDEO_PATH)
except RuntimeError as e:
    print(f"\n❌ Court auto-calibration failed:\n  {e}")
    print("   Check that the video shows clear court lines.")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)

# ── Video writer setup ────────────────────────────────────────────────────────
fps    = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
t_start = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx > 0 and frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            print(f"[{frame_idx}/{total_frames}] {frame_idx / elapsed:.1f} fps | "
                  f"score={engine.state.score} | {engine.state.rally_state}")

        landing = engine.process_frame(frame, frame_idx)
        score   = engine.state.score

        # ── Court boundary overlay ────────────────────────────────────────
        frame = engine.draw_court_boundaries(frame)

        # ── Score overlay ─────────────────────────────────────────────────
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

        # Rally state + last hitter debug overlay
        rs_color = (0, 200, 0) if engine.state.rally_state == "RALLY_ACTIVE" else (0, 140, 255)
        lh = engine.state.last_hitter_side or "?"
        cv2.putText(frame, f"{engine.state.rally_state} | hitter={lh}", (10, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rs_color, 1)

        serving = engine.state.serving_side
        serve_x = 85 if serving == 0 else 150
        cv2.circle(frame, (serve_x, 58), 5, (0, 255, 255), -1)

        if landing:
            color = (0, 220, 0) if landing.verdict.value == "IN" else (0, 0, 220)
            label = f"  {landing.verdict.value}  {landing.reason[:60]}"
            cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            print(
                f"[{frame_idx}] {landing.verdict.value} | "
                f"score={score} | src={landing.source} | {landing.reason}"
            )

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
