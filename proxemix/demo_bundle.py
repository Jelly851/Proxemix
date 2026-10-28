"""Stage 10 support — builds the demo bundle the UI serves: a small
synthetic (video, tracks, events) trio sized to match the synthetic
ingest test video, since Grand Central (used by every earlier stage) has
no underlying video file, only published annotations.

A "bundle" is any directory containing:
    video.mp4        - the footage the player shows
    proxemix.duckdb   - tracks/events/track_summary (+ summary tables),
                        populated by the same stage 3/4/5/6 pipeline every
                        other dataset in this package runs through
    bundle_meta.json  - {fps, width, height, duration_s}

`proxemix/serve.py --bundle <dir>` only reads that shape, so a real
footage bundle drops in later with zero UI changes - run this module's
`build_demo_bundle()` once to produce the synthetic one, or point at any
directory built the same way (see readme.md).
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

from . import storage, track_cleaning
from .calibration import apply_calibration_to_tracks, load_calibration_config
from .fact_sheet import write_fact_sheet
from .metrics_pipeline import run_metrics
from .report_verifier import verify_and_render
from .report_writer import StubReportWriter

FPS = 25.0
WIDTH, HEIGHT = 640, 480
DURATION_S = 12.0
PX_PER_M = 80.0  # both configs/calibration_demo.json and configs/zones_demo.json assume this

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def _track_specs(n_frames, fps):
    """Pixel-space (track_id, frame, x, y) generator for the 6 synthetic
    demo people. Each spec is a function of t (seconds, 0..DURATION_S)
    returning (x_px, y_px). See demo_bundle module docstring / readme.md
    for what each one is designed to exercise.
    """
    t = np.arange(n_frames) / fps

    def hold(x, y):
        return np.full(n_frames, x), np.full(n_frames, y)

    specs = {}
    # P1/P2: approach then stand close for most of the clip - exercises
    # contact episodes, Hall-band occupancy + one transition, and a
    # persistent DBSCAN group (needs the full-clip >=10s threshold).
    specs["P1_anchor"] = hold(320.0, 240.0)
    p2_y = np.where(t < 1.5, 40.0 + (170.0 - 40.0) * np.clip(t / 1.5, 0, 1), 170.0)
    specs["P2_approach"] = (np.full(n_frames, 320.0), p2_y)
    # P3: parked inside left_zone the whole clip - busiest-zone + fastest
    # zone-entry facts.
    specs["P3_leftzone_dwell"] = hold(120.0, 240.0)
    # P4: open floor, then walks into right_zone partway through - the
    # slow zone-entry contrast to P3, plus a distinct OD pair. Routed in
    # an L-shape (sideways first, then in) so its path never passes
    # within the group-detector's radius of the P1/P2 pair.
    p4_x = np.select(
        [t < 4.0, t < 6.0, t < 8.0],
        [320.0, 320.0 + (520.0 - 320.0) * np.clip((t - 4.0) / 2.0, 0, 1), 520.0],
        default=520.0,
    )
    p4_y = np.select(
        [t < 6.0, t < 8.0],
        [405.0, 405.0 + (240.0 - 405.0) * np.clip((t - 6.0) / 2.0, 0, 1)],
        default=240.0,
    )
    specs["P4_rightzone_entry"] = (p4_x, p4_y)
    # P5: fast straight crossing - the speed outlier.
    specs["P5_fast_crossing"] = (np.clip(t / DURATION_S, 0, 1) * 630.0 + 5.0, np.full(n_frames, 450.0))
    # P6: tight repeating circle, far from everyone else - near-zero net
    # displacement despite a long path, the directness-ratio outlier.
    specs["P6_jitter_circle"] = (
        560.0 + 40.0 * np.sin(2 * np.pi * t / 1.0),
        440.0 + 30.0 * np.cos(2 * np.pi * t / 1.0),
    )
    return specs, t


def make_synthetic_demo_tracks(fps=FPS, duration_s=DURATION_S):
    """Builds the canonical (track_id, frame, x, y) pixel table for the
    demo bundle's 6 synthetic people.
    """
    n_frames = int(round(duration_s * fps))
    specs, _ = _track_specs(n_frames, fps)
    frames = np.arange(n_frames)
    rows = []
    for track_id, (xs, ys) in specs.items():
        for f, x, y in zip(frames, xs, ys):
            rows.append((track_id, int(f), float(x), float(y)))
    df = pd.DataFrame(rows, columns=["track_id", "frame", "x", "y"])
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def ensure_demo_video(video_path, duration_s=DURATION_S, width=WIDTH, height=HEIGHT, fps=FPS):
    """Re-creates the demo clip (ffmpeg testsrc) if missing: a visible
    test pattern background the overlay draws on top of, standing in for
    real footage per the brief (Grand Central itself has no video file).
    """
    if os.path.exists(video_path):
        return video_path
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=duration={duration_s}:size={width}x{height}:rate={fps}",
        "-g", "12", "-bf", "0", "-movflags", "+faststart",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building demo video: {result.stderr[-2000:]}")
    return video_path


def build_demo_bundle(bundle_dir, force=False):
    """Builds the full demo bundle at `bundle_dir`: video.mp4,
    proxemix.duckdb (tracks/events/track_summary + summary tables,
    fact_sheet.json, report.md), bundle_meta.json. Idempotent unless
    force=True. Returns the bundle_meta dict.
    """
    os.makedirs(bundle_dir, exist_ok=True)
    video_path = os.path.join(bundle_dir, "video.mp4")
    db_path = os.path.join(bundle_dir, "proxemix.duckdb")
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")

    if not force and os.path.exists(meta_path) and os.path.exists(video_path) and os.path.exists(db_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ensure_demo_video(video_path)

    raw = make_synthetic_demo_tracks()
    cleaned, clean_report = track_cleaning.clean_tracks(raw, fps=FPS, sample_stride=1)

    if os.path.exists(db_path):
        os.remove(db_path)
    storage.store_cleaned_tracks(db_path, cleaned)

    cal_cfg = load_calibration_config(os.path.join(CONFIG_DIR, "calibration_demo.json"))
    calibrated = apply_calibration_to_tracks(cleaned, cal_cfg)
    zones_cfg_path = os.path.join(CONFIG_DIR, "zones_demo.json")

    con = storage.init_db(db_path)
    event_counts = run_metrics(
        con, calibrated, cal_cfg, zones_cfg_path,
        contact_threshold_m=1.5, contact_min_duration_s=2.0,
        pairwise_max_radius_m=3.6, od_grid_cells=2, outlier_top_n=10,
    )

    facts = write_fact_sheet(con, os.path.join(bundle_dir, "fact_sheet.json"))
    draft = StubReportWriter().draft(facts)
    clean_statements, dropped, markdown = verify_and_render(draft, facts, title="Proxemix Demo Report")
    with open(os.path.join(bundle_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(bundle_dir, "report_statements.json"), "w", encoding="utf-8") as f:
        json.dump(clean_statements, f, indent=2)
    con.close()

    meta = {
        "fps": FPS, "width": WIDTH, "height": HEIGHT, "duration_s": DURATION_S,
        "video_file": "video.mp4", "db_file": "proxemix.duckdb",
        "clean_report": clean_report, "event_counts": event_counts,
        "n_facts": len(facts), "n_dropped_poison_check": len(dropped),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
