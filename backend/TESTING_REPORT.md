# End-to-End Testing Report

**Date**: 2026-07-01
**Environment**: `backend/ubenv` (Python 3.10.11, CUDA 12.8)
**Test Scope**: Backend API, Scoring Engine, Court Calibration, Session Management

---

## ✅ Verified Working Components

### Dependencies (all installed in `ubenv`)
| Package | Version | Status |
|---------|---------|--------|
| torch | 2.11.0+cu128 | ✅ GPU available |
| torchvision | 0.26.0+cu128 | ✅ |
| ultralytics | 8.4.14 | ✅ |
| fastapi | 0.110.0 | ✅ |
| uvicorn | 0.27.1 | ✅ |
| opencv-python | 4.9.0.80 | ✅ |
| numpy | 1.26.4 | ✅ |

### Model Weights (present and loading)
- **YOLO-OBB**: `model2/models/claudette_trained/yolo_obb_finetune/weights/best.pt` (5.7 MB)
- **TrackNetV2**: `model2/models/claudette_trained/tracknetv2/tracknetv2_best.pth` (99 MB)

### Court Calibration
- Auto-detects corners from video using HSV segmentation + line refinement
- Fits homography to canonical BWF court (doubles boundary regardless of mode)
- Projects canonical lines for IN/OUT judgement
- **Tested**: Successfully calibrated on `merged_20260520_174744.mp4` (3190 frames)

### Scoring Engine Core Functionality
- Rally detection: sustained launch speed → `SERVE_PENDING` → `RALLY_ACTIVE`
- Hit detection: sharp direction change (>55°) at racket speed (>6 px/frame)
- Rally end: shuttle at rest low in frame OR vanished while descending near floor
- IN/OUT verdict: homography geometry against zone polygon (singles/doubles)
- **Unit tests**: All 3 pass (`test_scoring_corrections.py`)

### FastAPI Endpoints (13 routes loaded)
```
POST   /upload-video
POST   /process-video/{job_id}
GET    /job-status/{job_id}
GET    /results/{job_id}
GET    /download/{job_id}
POST   /live/start
POST   /live/chunk/{session_id}
POST   /live/finish/{session_id}
GET    /live/status/{session_id}
WS     /live/ws/{session_id}
```

### Session Management
- `ChunkSession` wraps `ScoringEngine` for chunked video processing
- Court calibrated once on first chunk, state carries across chunks
- Worker thread processes frames; async state broadcast via WebSocket/polling
- Annotated video writer runs on separate thread

---

## ⚠️ Critical Logical Bugs in Scoring Mechanism

### Bug 1: Players Don't Switch Sides After Game Ends
**File**: `llm/scoring_engine.py:452-457`
```python
def _game_over(self, winner: int):
    print(f"Game {self.state.game} won by player {winner + 1}: {self.state.score}")
    self.state.score = [0, 0]
    self.state.game += 1
    self.state.last_hitter_side = None
    self._prev_court_x = None
    # MISSING: side_to_player flip for next game
```
**Impact**: In real badminton, players switch ends after each game. The `side_to_player` mapping stays fixed, so LEFT/RIGHT court halves map to wrong players in subsequent games.

**Fix**:
```python
def _game_over(self, winner: int):
    # ... existing code ...
    # Flip sides for next game (players switch ends)
    left_player = self.state.side_to_player["LEFT"]
    right_player = self.state.side_to_player["RIGHT"]
    self.state.side_to_player = {"LEFT": right_player, "RIGHT": left_player}
```

---

### Bug 2: Verdict Correction Affects Wrong Game
**File**: `llm/scoring_engine.py:459-496`
- `pending_verdict` persists across game boundaries
- `correct_last_verdict()` modifies `state.score` directly without checking which game the verdict belongs to
- **Observed**: Correcting a Game 1 verdict during Game 2 produces negative scores (`[-1, 1]`)

**Root Cause**: No linkage between `PendingVerdict` and game number. The correction logic assumes current score reflects the game where the verdict occurred.

**Fix**:
```python
@dataclass
class PendingVerdict:
    landing_event: LandingEvent
    frame_idx: int
    is_correctable: bool = True
    game_number: int = 1  # NEW: track which game this verdict belongs to

def correct_last_verdict(self, new_verdict: Verdict) -> bool:
    if (self.state.pending_verdict is None 
        or not self.state.pending_verdict.is_correctable):
        return False
    
    # NEW: Ignore corrections from completed games
    if self.state.pending_verdict.game_number < self.state.game:
        self.state.pending_verdict.is_correctable = False
        return False
    
    # ... rest of correction logic ...
```

---

### Bug 3: Session Cleanup Race Condition
**File**: `session.py:304-309`
```python
def active_sessions() -> int:
    return sum(
        1 for s in SESSIONS.values()
        if s.latest_state["status"] not in ("complete", "error")
    )
```
- Worker thread updates `latest_state["status"]` asynchronously
- `active_sessions()` called from main thread may see stale status
- Finished sessions remain "active" until worker completes
- Blocks new sessions (VRAM limit: 1 at a time)

**Fix**:
```python
def active_sessions() -> int:
    return sum(
        1 for s in SESSIONS.values()
        if s.engine is not None  # Engine released in _finalize()
        and s.latest_state["status"] not in ("complete", "error")
    )
```

---

## 🔍 Minor Issues & Observations

### Detection Stride Behavior
- `DETECT_STRIDE=2` (default) runs shuttle inference every 2nd frame
- Hit detection still runs on interpolated positions between real detections
- May miss very fast hits if gap > `_HIT_GAP_MAX` (8 frames)

### Rally Abort Logic
- `_ABORT_FRAMES = 150` frames with no detection → abort rally, no point
- Could prematurely end long rallies with tracking gaps
- Consider making this configurable per game mode

### Cooldown Values (may need tuning)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `_HIT_COOLDOWN` | 4 frames | Min frames between hits |
| `_POINT_COOLDOWN` | 75 frames | Ignore shuttle pickup after point |
| `_GONE_FRAMES` | 10 frames | Wait before "vanished" rally end |

---

## 🧪 Test Commands for Future Regression Testing

```bash
# Activate environment
cd /root/Personal_Projects/Badminton_Analysis/game-tracker/backend
source ubenv/bin/activate

# Run unit tests
python -m pytest llm/test_scoring_corrections.py -v

# Test court calibration
python -c "
from llm.scoring_engine import ScoringEngine
engine = ScoringEngine(obb_weights='model2/models/claudette_trained/yolo_obb_finetune/weights/best.pt',
                       tracknet_weights='model2/models/claudette_trained/tracknetv2/tracknetv2_best.pth',
                       mode='singles', tracking_mode='hybrid', device='0')
engine.auto_calibrate('model2/dataset/TrackNetV2_Dataset/Professional/match3/video/merged_20260520_174744.mp4', n_frames=60)
print('Calibration OK')
"

# Test FastAPI loads
python -c "
from main import app
print('Routes:', [r.path for r in app.routes])
"

# Quick scoring engine test (100 frames)
python -c "
import cv2
from llm.scoring_engine import ScoringEngine
engine = ScoringEngine(obb_weights='model2/models/claudette_trained/yolo_obb_finetune/weights/best.pt',
                       tracknet_weights='model2/models/claudette_trained/tracknetv2/tracknetv2_best.pth',
                       mode='singles', tracking_mode='hybrid', device='0', detect_stride=2)
engine.auto_calibrate('model2/dataset/TrackNetV2_Dataset/Professional/match3/video/merged_20260520_174744.mp4', n_frames=30)
cap = cv2.VideoCapture('model2/dataset/TrackNetV2_Dataset/Professional/match3/video/merged_20260520_174744.mp4')
for i in range(100):
    ret, frame = cap.read()
    if not ret: break
    engine.process_frame(frame, i)
print(f'Score: {engine.state.score}, Landings: {len(engine.state.history)}')
cap.release()
"
```

---

## 📋 Recommended Fix Priority

| Priority | Issue | Effort | Risk if Unfixed |
|----------|-------|--------|-----------------|
| **P0** | Side switch after game | Low | Wrong player scoring in multi-game matches |
| **P0** | Verdict correction crosses games | Medium | Score corruption, negative scores |
| **P1** | Session cleanup race | Low | API blocks new sessions incorrectly |
| **P2** | Cooldown parameter tuning | Medium | Missed hits / false rally ends |

---

## Frontend Status
- **React Native / Expo** app at `frontend/myApp/`
- Live scoring screen with camera feed, WebSocket + polling fallback
- Video upload screen with progress polling and result playback
- API config: `API_BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8001'`
- **Ready for integration** once backend bugs fixed