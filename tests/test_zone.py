from src.analytics import ZoneMonitor, rectangle_polygon

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def make_track(track_id, x_center, y_bottom, label="person"):
    return {"track_id": track_id,
            "bbox": (x_center - 10, y_bottom - 50, x_center + 10, y_bottom),
            "label": label}


def test_dwell_time_uses_video_fps():
    zone = ZoneMonitor(SQUARE, fps=10)
    for _ in range(20):
        zone.update([make_track(1, 50, 50)])
    assert zone.is_inside(1)
    assert zone.dwell_seconds(1) == 2.0


def test_object_outside_zone_is_not_counted():
    zone = ZoneMonitor(SQUARE, fps=10)
    zone.update([make_track(1, 300, 50)])
    assert not zone.is_inside(1)
    assert zone.dwell_seconds(1) == 0.0


def test_rectangle_polygon_from_fractions():
    poly = rectangle_polygon(200, 100, (0.25, 0.5, 0.75, 1.0))
    assert poly == [(50, 50), (150, 50), (150, 100), (50, 100)]