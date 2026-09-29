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
    assert not any(_track_is_current(trk) for trk in tracker.active_tracks)
