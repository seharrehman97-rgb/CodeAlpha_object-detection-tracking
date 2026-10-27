import os
import tempfile

import altair as alt
import streamlit as st

from src.pipeline import Config, Pipeline

st.set_page_config(page_title="Vision Tracker", page_icon="🎯", layout="wide")

CLASS_OPTIONS = {"Person": 0, "Bicycle": 1, "Car": 2,
                 "Motorcycle": 3, "Bus": 5, "Truck": 7}

# Presets come from the benchmark: model size and input size vs speed.
PRESETS = {
    "Fast": {"model": "models/yolo11n.pt", "imgsz": 480, "width": 640},
    "Balanced": {"model": "models/yolo11n.pt", "imgsz": 640, "width": 960},
    "Accurate": {"model": "models/yolo11s.pt", "imgsz": 640, "width": 960},
}
# Rough CPU throughput (frames per second) per preset, for time estimates.
THROUGHPUT = {"Fast": 9.0, "Balanced": 5.0, "Accurate": 3.5}

from pathlib import Path

st.markdown(f"<style>{Path('assets/style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

hero_text, hero_art = st.columns([1.4, 1])
with hero_text:
    st.markdown(
        '<div class="kicker">Real-time object detection and tracking</div>'
        '<div class="hero-h1">Follow Every<br>'
        '<span class="nowrap"><span class="hl">Moving</span> Thing</span></div>'
        '<p class="hero-p">Upload a clip or open the webcam. Vision Tracker finds '
        'every person and vehicle, gives each one an identity, and tells you '
        'exactly when they appeared and when they left.</p>',
        unsafe_allow_html=True)
with hero_art:
    st.image("assets/lens.svg")

def run_analysis(mode, uploaded, seconds, preset, classes, conf, line_y,
                 show_trails, zone, min_duration, view):
    """Runs the pipeline, streams frames into the page, stores the results."""
    p = PRESETS[preset]
    cfg = Config(model=p["model"], conf=conf, width=p["width"],
                 classes=classes, tracker="histogram", imgsz=p["imgsz"],
                 line_y=line_y, trail=30 if show_trails else 0, zone=zone)

    tmp_path = None
    if mode == "Upload video":
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.close()
        tmp_path = tmp.name
        source = tmp_path
    else:
        source = 0  # the webcam of the computer running this app

    pipeline = Pipeline(cfg)
    bar = view.progress(0.0, text="Starting...")
    frame_slot = view.empty()
    last_frame = None

    try:
        frames = pipeline.run(source, max_seconds=seconds)
        for frame, info in frames:
            last_frame = frame
            if pipeline.is_live or not pipeline.total_frames:
                target = seconds
            else:
                target = min(seconds, pipeline.total_frames / pipeline.video_fps)
            target = max(target, 0.1)
            bar.progress(min(info["timestamp"] / target, 1.0),
                         text=f"Analyzing... {info['timestamp']:.1f}s of "
                              f"{target:.0f}s  |  {info['avg_fps']:.1f} FPS")
            if info["frame"] % 2 == 0:
                frame_slot.image(frame, channels="BGR")
        frames.close()
    except RuntimeError as err:  # for example: webcam or file could not be opened
        view.error(str(err))
        return
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    bar.progress(1.0, text="Done")
    if last_frame is not None:
        frame_slot.image(last_frame, channels="BGR")
    st.session_state["result"] = {
        "df": pipeline.log_dataframe(min_duration),
        "summary": pipeline.summary(),
        "last_frame": last_frame,
    }


# ---------------------------- Sidebar controls ----------------------------
with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Upload video", "Live webcam"], horizontal=True)
    uploaded = None
    if mode == "Upload video":
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    else:
        st.caption("The webcam works when the app runs on your own computer.")
    seconds = st.slider("Analyze for (seconds)", 5, 120, 15, step=5)

    st.header("Detection")
    preset = st.selectbox("Speed preset", list(PRESETS), index=1)
    chosen = st.multiselect("Classes to track", list(CLASS_OPTIONS),
                            default=["Person", "Car"])
    conf = st.slider("Confidence", 0.10, 0.90, 0.40, 0.05)
    if mode == "Upload video":
        est = seconds * 25 / THROUGHPUT[preset]
        st.caption(f"Estimated processing time on a CPU: about "
                   f"{est / 60:.1f} min" if est >= 60 else
                   f"Estimated processing time on a CPU: about {est:.0f} s")

    st.header("Analytics")
    line_y = st.slider("Counting line height", 0.1, 0.9, 0.6, 0.05)
    show_trails = st.checkbox("Show motion trails", value=True)
    use_zone = st.checkbox("Restricted zone", value=True)
    zone_x = st.slider("Zone: left to right", 0.0, 1.0, (0.3, 0.75), 0.05)
    zone_y = st.slider("Zone: top to bottom", 0.0, 1.0, (0.55, 1.0), 0.05)
    min_duration = st.slider("Hide objects seen less than (s)", 0.0, 5.0, 1.0, 0.5)

    run = st.button("Start analysis", type="primary")

# ------------------------------ Main area ---------------------------------
tab_live, tab_summary, tab_report = st.tabs(["Live view", "Summary", "Report"])

if run:
    if mode == "Upload video" and uploaded is None:
        tab_live.warning("Please upload a video first.")
    elif not chosen:
        tab_live.warning("Select at least one class to track.")
    else:
        zone = (zone_x[0], zone_y[0], zone_x[1], zone_y[1]) if use_zone else None
        run_analysis(mode, uploaded, seconds, preset,
                     [CLASS_OPTIONS[c] for c in chosen], conf, line_y,
                     show_trails, zone, min_duration, tab_live)

result = st.session_state.get("result")

with tab_live:
    if result is None and not run:
        st.info("Choose a source, adjust the settings in the sidebar, then "
                "press Start analysis.")
    elif result is not None and not run and result["last_frame"] is not None:
        st.image(result["last_frame"], channels="BGR",
                 caption="Last analyzed frame")

with tab_summary:
    if result is None:
        st.info("Results appear here after an analysis.")
    else:
        s, df = result["summary"], result["df"]
        people = int((df["class"] == "person").sum()) if len(df) else 0
        r1 = st.columns(4)
        r1[0].metric("Objects in report", len(df))
        r1[1].metric("People", people)
        r1[2].metric("Seconds analyzed", s["seconds_analyzed"])
        r1[3].metric("Average FPS", s["avg_fps"])
        r2 = st.columns(4)
        r2[0].metric("Crossed line (down)", sum(s["crossed_down"].values()))
        r2[1].metric("Crossed line (up)", sum(s["crossed_up"].values()))
        r2[2].metric("Entered restricted zone", s["ids_that_entered_zone"])
        r2[3].metric("IDs before filtering", s["unique_ids"])
        st.caption("Counts are approximate: the tracker can occasionally give "
                   "the same person a new ID after a long occlusion.")

with tab_report:
    if result is None:
        st.info("The report appears here after an analysis.")
    else:
        df = result["df"]
        if df.empty:
            st.warning("No objects stayed in view long enough. Try lowering "
                       "'Hide objects seen less than' or the confidence.")
        else:
            st.download_button("Download CSV report",
                               data=df.to_csv(index=False).encode("utf-8"),
                               file_name="analysis_report.csv",
                               mime="text/csv", type="primary")
            st.dataframe(df)
            st.subheader("Presence timeline")
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("first_seen_s:Q", title="Time (seconds)"),
                x2=alt.X2("last_seen_s"),
                y=alt.Y("track_id:N", title="Track ID",
                        sort=alt.EncodingSortField(field="first_seen_s", op="min")),
                color=alt.Color("class:N", title="Class"),
                tooltip=["track_id", "class", "appeared_at", "left_at",
                         "duration_s"],
            ).properties(height=max(200, 24 * len(df)))
            st.altair_chart(chart)