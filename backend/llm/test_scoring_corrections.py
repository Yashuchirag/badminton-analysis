"""
Test coverage for rally end and verdict correction fixes.
- Test 1: High lob (ascending trajectory) does not end rally
- Test 2: Clearly descending shuttle ends rally
- Test 3: Verdict correction updates score correctly
"""

import numpy as np
from scoring_engine import ScoringEngine, LandingEvent, Verdict, PendingVerdict


def test_high_lob_does_not_end_rally():
    """High lob (ascending trajectory) leaving frame should not end rally."""
    engine = ScoringEngine(mode="singles", device="cpu")

    # Simulate: shuttle going up and out of frame (lvy < DESCENDING_THRESHOLD)
    engine.state.rally_state = "RALLY_ACTIVE"
    engine._rally_start_frame = 0
    engine._last_det = (90, 300, 200, 0.8, "yolo", -2.5)  # ← vy = -2.5 (ascending)
    engine._floor_min_y = 400  # Court floor is at y >= 400

    # Move forward to GONE_FRAMES (should NOT trigger rally end because vy < threshold)
    landing = None
    for frame_idx in range(91, 91 + engine._GONE_FRAMES + 1):
        result = engine.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx)
        if result is not None:
            landing = result
            break

    # Should NOT end rally (no landing returned)
    assert landing is None, "High lob (ascending) should not end rally"
    assert engine.state.rally_state == "RALLY_ACTIVE", "Rally should still be active"
    print("✓ test_high_lob_does_not_end_rally passed")


def test_descending_shuttle_ends_rally():
    """Clearly descending shuttle leaving frame should end rally."""
    engine = ScoringEngine(mode="singles", device="cpu")

    # Set up a homography that maps pixel space to court space
    # Simple linear mapping: pixel (300, 420) → court (0.5, 2.0)
    # This avoids the sanity check rejection
    H = np.array([
        [0.001, 0.0, 0.0],      # cx = 0.001 * px
        [0.0, 0.001, 0.0],      # cy = 0.001 * py
        [0.0, 0.0, 1.0]         # homogeneous coord
    ], dtype=np.float32)
    engine.court.H = H

    # Simulate: shuttle clearly descending and low
    engine.state.rally_state = "RALLY_ACTIVE"
    engine._rally_start_frame = 0
    engine._last_det = (90, 300, 420, 0.8, "yolo", 3.0)  # ← vy = 3.0 (clearly descending)
    engine._floor_min_y = 400

    # Move forward to GONE_FRAMES (should trigger rally end at frame 100+lf=90 when lf-90=10)
    landing = None
    for frame_idx in range(91, 91 + engine._GONE_FRAMES + 1):
        result = engine.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx)
        if result is not None:
            landing = result
            break  # Landing was returned, exit loop

    # Should end rally and return landing event
    assert landing is not None, "Clearly descending shuttle should end rally"
    assert landing.verdict in (Verdict.IN, Verdict.OUT), "Should have a verdict"
    print("✓ test_descending_shuttle_ends_rally passed")


def test_verdict_correction_updates_score():
    """Correcting a verdict should reverse old point and award new point."""
    engine = ScoringEngine(mode="singles", device="cpu")

    # Start with 10-9 (player 0 ahead)
    engine.state.score = [10, 9]

    # Simulate an uncertain IN verdict (player 0 scored)
    landing = LandingEvent(
        frame_idx=100,
        pixel_x=320, pixel_y=450,
        court_x=0.5, court_y=2.0,
        confidence=0.45,  # Low confidence → uncertain
        source="yolo",
        near_line=True,
        verdict=Verdict.IN,
        reason="IN (near-line, low conf) — may be wrong"
    )
    engine.state.pending_verdict = PendingVerdict(
        landing_event=landing,
        frame_idx=100,
        is_correctable=True
    )

    # Correct to OUT (player 1 should score instead)
    success = engine.correct_last_verdict(Verdict.OUT)

    assert success is True, "Correction should succeed"
    assert engine.state.score == [9, 10], "Score should reverse and update to [9, 10]"
    assert landing.verdict == Verdict.OUT, "Landing event verdict should be updated"
    assert engine.state.pending_verdict.is_correctable is False, "Should mark as corrected"
    print("✓ test_verdict_correction_updates_score passed")


if __name__ == "__main__":
    test_high_lob_does_not_end_rally()
    test_descending_shuttle_ends_rally()
    test_verdict_correction_updates_score()
    print("✓ All tests passed")
