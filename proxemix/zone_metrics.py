"""Dwell per zone, zone visits, and latency to first entry — same boolean
predicate -> debounce -> RLE pattern, applied per-track to the "which zone
(if any)" column from proxemix/zones.py.
"""
import pandas as pd

from .groups import DEFAULT_GAP_STRIDE_FACTOR, DEFAULT_MIN_GAP_S, auto_gap_s


def detect_zone_visits(zoned_df, max_gap_s=None, debounce_frames=None,
                       min_gap_s=DEFAULT_MIN_GAP_S,
                       stride_factor=DEFAULT_GAP_STRIDE_FACTOR):
    """zoned_df: calibrated track rows with a 'zone' column (None/NaN where
    not in any zone). Returns one row per (track_id, zone) visit episode:
    track_id, zone, frame_start, frame_end, t_start, t_end, duration_s.

    **The gap tolerance is in SECONDS**, derived from the sampling stride when
    left as None (`groups.auto_gap_s`). The old `debounce_frames=40` default
    was the defect the SALSA validation found in the group filter, repaired
    here 2026-10-26; `debounce_frames` still means frames if passed.
    """
    df = zoned_df.dropna(subset=["zone"]).sort_values(["track_id", "frame"]).copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "track_id", "zone", "frame_start", "frame_end", "t_start", "t_end", "duration_s",
        ])
    grp = df.groupby("track_id", sort=False)
    if debounce_frames is not None:
        gap = grp["frame"].diff()
        tol = float(debounce_frames)
    else:
        gap = grp["t"].diff()
        tol = float(max_gap_s) if max_gap_s is not None else auto_gap_s(
            df, min_gap_s=min_gap_s, stride_factor=stride_factor)
    zone_changed = df["zone"] != grp["zone"].shift()
    new_episode = gap.isna() | (gap > tol) | zone_changed
    episode_id = new_episode.groupby(df["track_id"]).cumsum()
    df["episode_id"] = episode_id

    visits = df.groupby(["track_id", "episode_id"], as_index=False).agg(
        zone=("zone", "first"),
        frame_start=("frame", "min"),
        frame_end=("frame", "max"),
        t_start=("t", "min"),
        t_end=("t", "max"),
    )
    visits["duration_s"] = visits["t_end"] - visits["t_start"]
    return visits.drop(columns=["episode_id"]).sort_values(["track_id", "frame_start"]).reset_index(drop=True)


def latency_to_first_entry(visits, track_first_t):
    """visits: output of detect_zone_visits. track_first_t: {track_id: t of
    that track's first row in the full (unzoned) track table} — latency is
    measured from when the track is first observed at all, not from t=0.
    Returns one row per (track_id, zone): first_entry_t, latency_s.
    """
    first = visits.sort_values(["track_id", "zone", "t_start"]).groupby(
        ["track_id", "zone"], as_index=False
    ).first()[["track_id", "zone", "frame_start", "t_start"]]
    first["origin_t"] = first["track_id"].map(track_first_t)
    first["latency_s"] = first["t_start"] - first["origin_t"]
    return first


def visits_to_event_rows(visits):
    rows = []
    for _, r in visits.iterrows():
        rows.append({
            "type": "zone_dwell", "actor_ids": r.track_id,
            "frame_start": int(r.frame_start), "frame_end": int(r.frame_end),
            "zone": r.zone, "value": float(r.duration_s),
        })
    return pd.DataFrame(rows)


def latency_to_event_rows(latency_df):
    rows = []
    for _, r in latency_df.iterrows():
        rows.append({
            "type": "zone_first_entry", "actor_ids": r.track_id,
            "frame_start": int(r.frame_start), "frame_end": int(r.frame_start),
            "zone": r.zone, "value": float(r.latency_s),
        })
    return pd.DataFrame(rows)
