from src.analytics import LineCounter, TrailManager


def make_track(track_id, y_bottom, label="person"):
    return {"track_id": track_id,
            "bbox": (0, y_bottom - 100, 50, y_bottom),
            "label": label}


def test_counts_downward_crossing():
    counter = LineCounter(line_y=100)
    counter.update([make_track(1, 80)])
    counter.update([make_track(1, 120)])
    assert counter.down["person"] == 1
    assert counter.up["person"] == 0


def test_counts_upward_crossing():
    counter = LineCounter(line_y=100)
    counter.update([make_track(1, 120)])
    counter.update([make_track(1, 80)])
    assert counter.up["person"] == 1
    assert counter.down["person"] == 0


def test_object_is_counted_only_once():
    counter = LineCounter(line_y=100)
    for y in (80, 120, 80, 120):
        counter.update([make_track(1, y)])
    assert sum(counter.down.values()) + sum(counter.up.values()) == 1


def test_no_crossing_means_no_count():
    counter = LineCounter(line_y=100)
    counter.update([make_track(1, 50)])
    counter.update([make_track(1, 70)])
    assert sum(counter.down.values()) == 0
    assert sum(counter.up.values()) == 0


def test_trail_length_is_limited():
    trails = TrailManager(length=5)
    for i in range(10):
        trails.update([make_track(1, 100 + i)])
    assert len(trails.trails[1]) == 5