"""Stage 3 — track cleaning, per the build spec:

* classify track births/deaths as border (real entry/exit) vs mid-frame
  (occlusion artifact, i.e. SAM/annotator lost and re-assigned the ID)
* bridge mid-frame gaps under ~2s via constant-velocity gating +
  scipy.optimize.linear_sum_assignment
* interpolate the bridged + any internal gaps
* Savitzky-Golay smooth the position series
* finite-difference velocities

Operates on the canonical (track_id, frame, x, y) table produced by any
stage-1/2 adapter (here, the Grand Central adapter standing in for SAM).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

BORDER_MARGIN_PX = 30
FRAME_W, FRAME_H = 1920, 1080
MAX_GAP_SECONDS = 2.0


def _is_border(x, y, w=FRAME_W, h=FRAME_H, margin=BORDER_MARGIN_PX):
    return x <= margin or x >= w - margin or y <= margin or y >= h - margin


def classify_endpoints(df):
    """One row per track: birth/death frame, position, velocity at death
    (from its last two points, 0 if the track is a single point), and
    border/mid_frame classification for both ends.
    """
    rows = []
    for track_id, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        bx, by, bf = g.iloc[0][["x", "y", "frame"]]
        dx_, dy_, df_ = g.iloc[-1][["x", "y", "frame"]]
        if len(g) >= 2:
            p0, p1 = g.iloc[-2], g.iloc[-1]
            dt = p1["frame"] - p0["frame"]
            vx = (p1["x"] - p0["x"]) / dt if dt else 0.0
            vy = (p1["y"] - p0["y"]) / dt if dt else 0.0
        else:
            vx, vy = 0.0, 0.0
        rows.append({
            "track_id": track_id,
            "birth_frame": int(bf), "birth_x": float(bx), "birth_y": float(by),
            "birth_type": "border" if _is_border(bx, by) else "mid_frame",
            "death_frame": int(df_), "death_x": float(dx_), "death_y": float(dy_),
            "death_vx": float(vx), "death_vy": float(vy),
            "death_type": "border" if _is_border(dx_, dy_) else "mid_frame",
            "n_points": len(g),
        })
    return pd.DataFrame(rows)


def bridge_tracks(df, fps=25.0, max_gap_seconds=MAX_GAP_SECONDS, max_speed_px_per_frame=None):
    """Stitch mid-frame deaths to mid-frame births that occur shortly after,
    at a plausible position given constant-velocity extrapolation.

    Returns (remapped_df, merges) where merges is a list of
    (dead_track_id, reborn_track_id, gap_frames, cost) for every bridge made.
    Candidates are bucketed by time so each local assignment problem stays
    small; linear_sum_assignment resolves which death matches which birth
    whenever more than one candidate competes for the same window.
    """
    endpoints = classify_endpoints(df)
    max_gap_frames = max_gap_seconds * fps

    if max_speed_px_per_frame is None:
        # data-driven gate: 95th percentile of observed per-step speed,
        # scaled up a little as slack for the extrapolation error.
        steps = []
        for _, g in df.groupby("track_id", sort=False):
            g = g.sort_values("frame")
            if len(g) < 2:
                continue
            d = np.diff(g[["x", "y"]].values, axis=0)
            dt = np.diff(g["frame"].values)
            dt[dt == 0] = 1
            steps.append(np.hypot(d[:, 0], d[:, 1]) / dt)
        all_speeds = np.concatenate(steps)
        max_speed_px_per_frame = np.percentile(all_speeds, 95) * 1.5

    deaths = endpoints[endpoints.death_type == "mid_frame"].sort_values("death_frame")
    births = endpoints[endpoints.birth_type == "mid_frame"].sort_values("birth_frame")

    bucket_size = max(1, int(np.ceil(max_gap_frames)))
    deaths = deaths.assign(_bucket=(deaths.death_frame // bucket_size).astype(int))
    births = births.assign(_bucket=(births.birth_frame // bucket_size).astype(int))
    deaths_by_bucket = {b: g for b, g in deaths.groupby("_bucket")}

    merges = []
    claimed_deaths = set()
    claimed_births = set()

    for bucket, birth_group in births.groupby("_bucket"):
        candidate_deaths = pd.concat(
            [deaths_by_bucket.get(bucket - 1, deaths.iloc[0:0]), deaths_by_bucket.get(bucket, deaths.iloc[0:0])]
        )
        candidate_deaths = candidate_deaths[~candidate_deaths.track_id.isin(claimed_deaths)]
        birth_group = birth_group[~birth_group.track_id.isin(claimed_births)]
        if candidate_deaths.empty or birth_group.empty:
            continue

        n_d, n_b = len(candidate_deaths), len(birth_group)
        cost = np.full((n_d, n_b), 1e9)
        d_rows = candidate_deaths.reset_index(drop=True)
        b_rows = birth_group.reset_index(drop=True)
        for i, d in d_rows.iterrows():
            for j, b in b_rows.iterrows():
                gap = b["birth_frame"] - d["death_frame"]
                if gap <= 0 or gap > max_gap_frames:
                    continue
                pred_x = d["death_x"] + d["death_vx"] * gap
                pred_y = d["death_y"] + d["death_vy"] * gap
                dist = np.hypot(pred_x - b["birth_x"], pred_y - b["birth_y"])
                if dist <= max_speed_px_per_frame * gap:
                    cost[i, j] = dist

        if not np.any(cost < 1e9):
            continue
        row_idx, col_idx = linear_sum_assignment(cost)
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] >= 1e9:
                continue
            d = d_rows.iloc[i]
            b = b_rows.iloc[j]
            merges.append((d["track_id"], b["track_id"], int(b["birth_frame"] - d["death_frame"]), float(cost[i, j])))
            claimed_deaths.add(d["track_id"])
            claimed_births.add(b["track_id"])

    # union-find style remap: chase chains (a death bridges to a birth,
    # whose own death may bridge further) down to one canonical id.
    parent = {}

    def find(tid):
        while parent.get(tid, tid) != tid:
            tid = parent[tid]
        return tid

    for dead, reborn, _, _ in merges:
        parent[find(reborn)] = find(dead)

    remap = {tid: find(tid) for tid in df.track_id.unique()}
    out = df.copy()
    out["track_id"] = out["track_id"].map(remap)
    return out, merges


def interpolate_and_smooth(df, fps=25.0, sample_stride=20, savgol_window=11, savgol_poly=2):
    """For each (post-merge) track: fill gaps on the data's own sample
    grid via linear interpolation, Savitzky-Golay smooth x/y, then
    finite-difference velocity. Adds columns t, vx, vy, interpolated.
    """
    cleaned = []
    for track_id, g in df.groupby("track_id", sort=False):
        g = g.drop_duplicates(subset="frame").sort_values("frame")
        frames_observed = g["frame"].values
        full_frames = np.arange(frames_observed.min(), frames_observed.max() + 1, sample_stride)
        x_full = np.interp(full_frames, frames_observed, g["x"].values)
        y_full = np.interp(full_frames, frames_observed, g["y"].values)
        interpolated = ~np.isin(full_frames, frames_observed)

        n = len(full_frames)
        if n >= savgol_poly + 2:
            w = min(savgol_window, n if n % 2 == 1 else n - 1)
            if w > savgol_poly:
                x_smooth = savgol_filter(x_full, window_length=w, polyorder=savgol_poly)
                y_smooth = savgol_filter(y_full, window_length=w, polyorder=savgol_poly)
            else:
                x_smooth, y_smooth = x_full, y_full
        else:
            x_smooth, y_smooth = x_full, y_full

        t = full_frames / fps
        if n >= 2:
            vx = np.gradient(x_smooth, t)
            vy = np.gradient(y_smooth, t)
        else:
            vx = np.zeros(n)
            vy = np.zeros(n)

        cleaned.append(pd.DataFrame({
            "track_id": track_id, "frame": full_frames, "t": t,
            "x": x_smooth, "y": y_smooth, "vx": vx, "vy": vy,
            "interpolated": interpolated,
        }))

    return pd.concat(cleaned, ignore_index=True)


def clean_tracks(df, fps=25.0, max_gap_seconds=MAX_GAP_SECONDS, sample_stride=20):
    """Full stage-3 pipeline. Returns (cleaned_df, report) where report
    carries the before/after counts the build brief asks to be proven.
    """
    tracks_before = df.track_id.nunique()
    endpoints = classify_endpoints(df)
    bridged_df, merges = bridge_tracks(df, fps=fps, max_gap_seconds=max_gap_seconds)
    tracks_after_bridge = bridged_df.track_id.nunique()
    cleaned = interpolate_and_smooth(bridged_df, fps=fps, sample_stride=sample_stride)

    report = {
        "tracks_before": tracks_before,
        "births_border": int((endpoints.birth_type == "border").sum()),
        "births_mid_frame": int((endpoints.birth_type == "mid_frame").sum()),
        "deaths_border": int((endpoints.death_type == "border").sum()),
        "deaths_mid_frame": int((endpoints.death_type == "mid_frame").sum()),
        "gaps_bridged": len(merges),
        "tracks_after": tracks_after_bridge,
        "rows_before": len(df),
        "rows_after_interp": len(cleaned),
        "rows_interpolated": int(cleaned["interpolated"].sum()),
    }
    return cleaned, report
