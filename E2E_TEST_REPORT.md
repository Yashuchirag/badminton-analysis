# End-to-End Test Report

Date: 2026-07-08
Scope: full frontend and backend flow, with a deep audit of the scoring engine logic.
No project source files were modified. All test files live in the project root.

## Deliverables

| File | What it covers | Result |
|---|---|---|
| `test_scoring_engine_logic.py` | Scoring engine state machine, hit detection, landing verdicts, 21-point rules, deuce, game transitions, corrections | 19 passed, 4 xfailed |
| `test_backend_api_e2e.py` | Real FastAPI app + real ChunkSession worker over HTTP and WebSocket, upload flow, live flow, error paths, frontend contract fields | 7 passed, 1 xfailed |

Run both with:

```
backend/ubenv/bin/python -m pytest test_scoring_engine_logic.py test_backend_api_e2e.py -rax
```

Combined result: **26 passed, 5 xfailed**. Every `test_flaw_*` test is marked `xfail(strict=True)`: it encodes the correct behaviour and currently fails, so it documents a live bug. Once a bug is fixed, the test will flip to an error until the marker is removed, which makes each fix self-verifying.

The scoring tests replace only the YOLO player tracker and the shuttle detection feed with scripted fakes and use a synthetic homography, so every branch of the real `ScoringEngine` logic runs. The API tests replace only `ScoringEngine` with a fake; upload handling, session threading, chunk sequencing, broadcasts, file cleanup, and the WebSocket all run production code.

Frontend checks: `npx tsc --noEmit` passes cleanly. `npx expo lint` reports 0 errors and 4 pre-existing `react-hooks/exhaustive-deps` warnings.

A real-pipeline smoke test also ran on GPU (CUDA) with the trained OBB + TrackNetV2 weights against the first 500 frames of `backend/verified.mp4`. Calibration succeeded and frames processed end to end.

---

## Confirmed scoring engine flaws (each has an xfail test)

### 1. Floor impact counts as a racket hit, so OUT points go to the wrong player
`_detect_hit` in `backend/llm/scoring_engine.py` has no height or floor gate. The sharp trajectory kink when the shuttle strikes the floor satisfies the same angle-and-speed test as a real racket hit, so the engine logs "Hit by LEFT" at the moment of landing and flips `last_hitter_side`. An OUT verdict then awards the point to the actual hitter instead of the receiver.
Test: `test_flaw_landing_impact_must_not_count_as_hit`.
Corroboration: on real footage the smoke test logged hits with turn speeds up to 450 px/frame, which is physically impossible for a tracked shuttle. There is no maximum-speed sanity check either, so detection noise (jumps between two different objects) also registers as hits. On real video, hitter attribution is effectively unreliable.

### 2. A correction that reaches 21 does not end the game
`correct_last_verdict` adjusts the score but never calls `_check_serve_change`, so a correction that brings a player to 21 (or 30) leaves the game open at, for example, 21-19.
Test: `test_flaw_correction_reaching_21_should_end_game`.

### 3. Corrections never move the serve
In badminton the scorer serves next. `correct_last_verdict` moves the point but leaves `serving_side` unchanged, so after a correction the wrong player is shown as serving and subsequent serve logic runs from the wrong side.
Test: `test_flaw_correction_should_move_serve_to_new_scorer`.

### 4. Corrections use the current side mapping, not the mapping at landing time
`correct_last_verdict` resolves players through `side_to_player` as it is now. If the mapping drifted between the landing and the correction (tracker noise, or the mid-game ends change in game 3 at 11 points), the reversal credits the wrong player.
Test: `test_flaw_correction_should_use_side_mapping_from_landing_time`.

## Design gaps documented by passing tests

### 5. A game-deciding uncertain verdict can never be corrected
When an uncertain landing ends the game, `_game_over` advances the game counter immediately, and the cross-game guard in `correct_last_verdict` (`pending.game_number < state.game`) then rejects the correction. The most important call of the game is the one call that is permanently frozen.
Test: `test_game_deciding_uncertain_verdict_cannot_be_corrected` (passing, documents current behaviour).

### 6. Discarded far-out landings still pollute the results
`_finish_rally` discards landings beyond `_MAX_PLAUSIBLE_OUT` (no point is scored, verdict stays PENDING), but the event is still appended to history and returned, so the session layer records it in `landing_events`. The frontend's landing count includes phantom events.
Test: `test_far_outside_landing_discarded` (passing, documents current behaviour).

## Scoring engine, report-only observations

7. After `_game_over` the score resets to [0, 0] and no per-game history is kept. The final results payload therefore shows 0-0 after a completed game, and the "Final score" alert in `index.tsx` is misleading.
8. `_verdict_to_scorer_side` treats every non-IN verdict as OUT, including LET. A let should replay the rally, not score.
9. Service-court rules are not modeled at all (right or left service box by even or odd score, service faults).
10. No API endpoint exposes `correct_last_verdict`, so the frontend has no way to trigger a correction. The whole correction subsystem is unreachable from the app.

## Backend API and session findings

### 11. An abandoned live session blocks the server forever (xfail test)
`session.cleanup_idle` is only called inside `create_session`, but every session-creating endpoint returns 409 while any session is active. If a live session is abandoned (phone dies, app killed, finish never sent), `active_sessions()` stays at 1 forever, every new upload and live start is refused, and the cleanup that would evict the stale session can never run. Only a server restart recovers.
Test: `test_flaw_stale_abandoned_live_session_should_be_evicted`.

### 12. A missing chunk stalls live scoring until the session is stopped
The session worker waits indefinitely for the next sequence number and only skips gaps after `finish()`. So if one chunk never arrives, scoring freezes (score stops updating, backlog grows) for the rest of the session. Combined with frontend finding 15 below, a single failed chunk upload does exactly this in practice.
Test: `test_live_missing_chunk_stalls_until_finish_then_gap_is_skipped` (passing, documents current behaviour).

13. Minor: the "Job is already processing" 400 branch in `backend/main.py` (line 83) is unreachable, because `process_video` pops the job from `uploads` and the 404 membership check runs first. A double process observably returns 404.

## Frontend findings (file and line references)

14. `frontend/myApp/app/live.tsx:141` and the upload call in `app/index.tsx` set `'Content-Type': 'multipart/form-data'` manually without a boundary parameter. On web this breaks the upload outright (the server cannot parse the body); on native it works only because React Native's fetch replaces the header. The header should be omitted so fetch sets it with the boundary.
15. `frontend/myApp/app/live.tsx:146`: `FileSystem.deleteAsync(item.uri)` runs in a `finally` block, so the local chunk file is deleted even when its upload failed. The chunk can never be retried, which triggers backend finding 12: one transient network error freezes live scoring for the remainder of the session.
16. `frontend/myApp/app/index.tsx:45`: the status poller only handles `processing`, `calibrating`, `complete`, and `error`. The backend also emits `starting` and `loading`, during which the progress UI sits idle. Cosmetic.
17. `app/index.tsx` `chooseVideoSource` uses `prompt()`, which only exists on web; on a phone the dialog never appears.
18. The `mode` query parameter (singles or doubles) is never sent on upload, so the backend always scores in singles mode.

## Real-pipeline smoke test (GPU)

Setup: real `ScoringEngine` with the trained weights (`yolo_obb_finetune/weights/best.pt`, `tracknetv2_best.pth`), device cuda, `detect_stride=2`, first 500 frames of `backend/verified.mp4` (30 fps, 1280x720).

Results:
- Auto-calibration succeeded and court coordinates flowed through the pipeline.
- Throughput was **4.3 fps**, roughly 7x slower than real time. Live mode therefore cannot keep up: the phone produces 30 fps chunks while the server consumes 4.3 fps, so the backlog grows without bound and the displayed score lags further behind every second. A 5-minute recorded video takes about 35 minutes to process.
- Hit detection fired every 6 to 16 frames with turn speeds up to 450 px/frame (see flaw 1). Zero landing events and no score change occurred in 16.7 seconds of real rally play, so rally-end detection on real footage deserves scrutiny beyond this one clip.

## Proposed fixes (awaiting confirmation, nothing applied)

A. Gate `_detect_hit` on shuttle height (reject kinks near `_floor_min_y`) and add a maximum plausible speed, fixing flaw 1 and the noise-hit spam.
B. In `correct_last_verdict`: call `_check_serve_change` after adjusting the score, update `serving_side` to the corrected scorer, and snapshot `side_to_player` into `PendingVerdict` at landing time. Fixes flaws 2, 3, and 4.
C. Call `cleanup_idle()` at the top of `process_video` and `live_start` before the active-session checks (or run a periodic reaper), fixing flaw 11.
D. `live.tsx`: drop the manual multipart header, and delete the chunk file only after a successful upload, keeping failed chunks for retry. Fixes findings 14 and 15 and defuses 12.
E. `index.tsx`: drop the manual multipart header, send the `mode` parameter, handle `starting` and `loading` in the poller, and replace `prompt()` with a native-friendly chooser. Fixes 16, 17, 18.
F. Keep a per-game score history in `MatchState` so final results survive `_game_over`. Fixes 7.
G. Decide the product behaviour for 5, 6, 8, 9, and 10 (correction endpoint, LET handling, service courts); these need direction before code.
