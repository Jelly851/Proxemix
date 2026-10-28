"""Grand Central adapter — loads the published Grand Central Station
pedestrian trajectories (OpenTraj format) into the canonical track table
(track_id, frame, x, y) that stands in for SAM output until the real
Meta API key arrives.

Source format (one .txt file per track, under datasets/GC/Annotation/):
one float per line, in repeating triples (x, y, frame). x/y are pixel
coordinates in the 1920x1080 reference frame; frame is the absolute
frame number in the source video, sampled every 20 frames (0.8s at the
dataset's native 25 fps).
"""
import glob
import os

import numpy as np
import pandas as pd

FRAME_W, FRAME_H = 1920, 1080


def load_grand_central_tracks(annotation_dir):
    """Return a DataFrame with columns track_id, frame, x, y, one row per
    annotated point, sorted by track_id then frame.
    """
    files = sorted(glob.glob(os.path.join(annotation_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"no .txt annotation files found under {annotation_dir}")

    rows = []
    for fp in files:
        track_id = os.path.splitext(os.path.basename(fp))[0]
        vals = np.loadtxt(fp)
        if vals.size == 0:
            continue
        vals = vals.reshape(-1, 3)
        for x, y, frame in vals:
            rows.append((track_id, int(round(frame)), float(x), float(y)))

    df = pd.DataFrame(rows, columns=["track_id", "frame", "x", "y"])
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    return df


def make_synthetic_fixture(out_dir, n_tracks=8, seed=0):
    """Write a tiny set of .txt files in the exact OpenTraj/Grand Central
    format (one flat value per line, repeating x, y, frame triples) so the
    adapter can be proven even if the real annotation files are absent.
    Mirrors the real data's conventions: 25 fps, 20-frame sample stride,
    1920x1080 frame, and a few short/mid-frame-gap tracks for variety.
    """
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_tracks):
        length = rng.integers(3, 20)
        start_frame = int(rng.integers(0, 5000))
        x0, y0 = rng.uniform(0, 1920), rng.uniform(0, 1080)
        vx, vy = rng.uniform(-5, 5), rng.uniform(-5, 5)
        lines = []
        for k in range(length):
            frame = start_frame + k * 20
            x = np.clip(x0 + vx * k, 0, 1919)
            y = np.clip(y0 + vy * k, 0, 1079)
            lines += [f"{x:.2f}", f"{y:.2f}", f"{frame}"]
        with open(os.path.join(out_dir, f"{i:06d}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    return out_dir
