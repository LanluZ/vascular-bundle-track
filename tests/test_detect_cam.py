import numpy as np
import pytest

from detect_cam import _create_tracker, _track_box, _track_is_current


@pytest.mark.parametrize("name", ["ocsort", "bytetrack"])
def test_create_tracker_returns_boxmot_backend(name):
    tracker = _create_tracker(name)
    assert hasattr(tracker, "update")
    assert hasattr(tracker, "active_tracks")


def _feed(tracker, dets, image):
    tracker.update(np.asarray(dets, dtype=np.float32).reshape(-1, 6), image)


@pytest.mark.parametrize("name", ["ocsort", "bytetrack"])
def test_track_box_exposes_xyxy_for_both_backends(name):
    tracker = _create_tracker(name)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for step in range(3):
        x = 10.0 + 6.0 * step
        _feed(tracker, [[x, x, x + 40, x + 40, 0.9, 0]], image)
    assert tracker.active_tracks, "synthetic detections should produce a track"
    box = np.asarray(_track_box(tracker.active_tracks[0]), dtype=float).reshape(-1)
    assert box.shape == (4,)
    assert np.all(np.isfinite(box))
    assert box[2] > box[0] and box[3] > box[1]


@pytest.mark.parametrize("name", ["ocsort", "bytetrack"])
def test_lost_tracks_are_skipped(name):
    tracker = _create_tracker(name)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for step in range(3):
        x = 10.0 + 6.0 * step
        _feed(tracker, [[x, x, x + 40, x + 40, 0.9, 0]], image)
    _feed(tracker, [], image)  # one frame without detections
    assert not any(_track_is_current(trk, tracker) for trk in tracker.active_tracks)


def test_ocsort_gate_skips_new_track_created_after_warmup():
    """Replicates boxmot's OcSort output gate (ocsort.py:380-383).

    A track is output only when ``time_since_update < 1`` AND
    (``hit_streak >= min_hits`` OR ``frame_count <= min_hits``); ``min_hits``
    (default 3) and ``frame_count`` live on the tracker, not the track.

    A track created after the warm-up window has ``time_since_update == 0``
    but ``hit_streak == 0 < min_hits`` while ``frame_count > min_hits`` -- it
    must be skipped even though it was updated this frame.
    """
    tracker = _create_tracker("ocsort")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):  # frames 1-3: warm-up passes with no detections
        _feed(tracker, [], image)
    _feed(tracker, [[10.0, 10.0, 50.0, 50.0, 0.9, 0]], image)  # frame 4: new track

    assert len(tracker.active_tracks) == 1, "track should be created"
    trk = tracker.active_tracks[0]
    # Preconditions: updated this frame, but below min_hits and past warm-up.
    assert int(trk.time_since_update) < 1
    assert int(trk.hit_streak) < tracker.min_hits
    assert tracker.frame_count > tracker.min_hits
    assert not _track_is_current(trk, tracker)


def test_ocsort_gate_outputs_via_warmup_then_hit_streak():
    """The two accepting branches of boxmot's gate: a brand-new track
    (``hit_streak == 0``) is output during warm-up, and once ``frame_count``
    passes ``min_hits`` output continues only via ``hit_streak >= min_hits``.
    """
    tracker = _create_tracker("ocsort")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _feed(tracker, [[10.0, 10.0, 50.0, 50.0, 0.9, 0]], image)  # frame 1: brand-new

    trk = tracker.active_tracks[0]
    assert int(trk.hit_streak) < tracker.min_hits
    assert tracker.frame_count <= tracker.min_hits
    assert _track_is_current(trk, tracker)  # warm-up branch

    for _ in range(3):  # frames 2-4: consecutive hits lift hit_streak to 3
        _feed(tracker, [[10.0, 10.0, 50.0, 50.0, 0.9, 0]], image)

    trk = tracker.active_tracks[0]
    assert int(trk.time_since_update) < 1
    assert int(trk.hit_streak) >= tracker.min_hits
    assert tracker.frame_count > tracker.min_hits
    assert _track_is_current(trk, tracker)  # hit_streak branch
