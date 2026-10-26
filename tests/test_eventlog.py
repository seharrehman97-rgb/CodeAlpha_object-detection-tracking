from src.analytics import EventLog, format_time


def make_track(track_id, label="person"):
    return {"track_id": track_id, "bbox": (0, 0, 10, 10), "label": label}


def test_format_time():
    assert format_time(0) == "00:00"
    assert format_time(65.4) == "01:05"


def test_event_log_records_appear_and_leave():
    log = EventLog()
    for i in range(30):
        log.update(i * 0.1, [make_track(1)])  # from 0.0 s to 2.9 s
    row = log.to_dataframe().iloc[0]
    assert row["appeared_at"] == "00:00"
    assert row["left_at"] == "00:02"
    assert row["duration_s"] == 2.9


def test_short_tracks_are_filtered_out():
    log = EventLog()
    log.update(0.0, [make_track(1)])
    log.update(0.2, [make_track(1)])  # object 1 lived only 0.2 s
    for i in range(30):
        log.update(i * 0.1, [make_track(2)])  # object 2 lived 2.9 s
    df = log.to_dataframe(min_duration=1.0)
    assert list(df["track_id"]) == [2]
    