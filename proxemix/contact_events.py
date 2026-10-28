"""Contact/proximity episodes per dyad — distance < threshold sustained
>= min_duration (COVID-literature convention: 1.5m / 60s), both
configurable. Pattern: boolean predicate per frame -> debounce -> RLE ->
event rows, applied to the shared pairwise-distance stream from
proxemix/pairwise.py.

Debounce here means: consecutive sub-threshold detections for the same
dyad merge into one episode as long as the gap between them is within
tolerance (tolerating brief tracking flicker); a bigger gap starts a new
episode.

**The gap tolerance is in SECONDS**, derived from the data's own sampling
stride when left as None — see `groups.auto_gap_s`. A fixed frame count is
only meaningful at the stride it was chosen for, which is the defect the
SALSA validation found in the group filter on 2026-10-25; the same 40-frame
default was sitting here, in `hall_bands.py` and in `zone_metrics.py`, so all
three were repaired the same way on 2026-10-26.
"""
import numpy as np
import pandas as pd

from .groups import DEFAULT_GAP_STRIDE_FACTOR, DEFAULT_MIN_GAP_S, auto_gap_s


def detect_contact_episodes(pairwise_df, distance_threshold_m=1.5, min_duration_s=60.0,
                             max_gap_s=None, debounce_frames=None,
                             min_gap_s=DEFAULT_MIN_GAP_S,
                             stride_factor=DEFAULT_GAP_STRIDE_FACTOR):
    hits = pairwise_df[pairwise_df["distance_m"] < distance_threshold_m].copy()
    if hits.empty:
        return pd.DataFrame(columns=[
            "track_a", "track_b", "frame_start", "frame_end", "t_start", "t_end",
            "duration_s", "n_samples", "mean_distance_m",
        ])
    hits = hits.sort_values(["track_a", "track_b", "frame"])
    grp = hits.groupby(["track_a", "track_b"], sort=False)
    if debounce_frames is not None:
        gap = grp["frame"].diff()
        tol = float(debounce_frames)
    else:
        gap = grp["t"].diff()
        tol = float(max_gap_s) if max_gap_s is not None else auto_gap_s(
            hits, min_gap_s=min_gap_s, stride_factor=stride_factor)
    new_episode = gap.isna() | (gap > tol)
    episode_id = new_episode.groupby([hits["track_a"], hits["track_b"]]).cumsum()

    hits["episode_id"] = episode_id
    agg = hits.groupby(["track_a", "track_b", "episode_id"]).agg(
        frame_start=("frame", "min"),
        frame_end=("frame", "max"),
        t_start=("t", "min"),
        t_end=("t", "max"),
        n_samples=("frame", "size"),
        mean_distance_m=("distance_m", "mean"),
    ).reset_index(drop=False)
    agg["duration_s"] = agg["t_end"] - agg["t_start"]
    episodes = agg[agg["duration_s"] >= min_duration_s].drop(columns=["episode_id"])
    return episodes.sort_values("duration_s", ascending=False).reset_index(drop=True)


def episodes_to_event_rows(episodes, event_type="contact"):
    rows = []
    for _, r in episodes.iterrows():
        rows.append({
            "type": event_type,
            "actor_ids": f"{r.track_a},{r.track_b}",
            "frame_start": int(r.frame_start),
            "frame_end": int(r.frame_end),
            "zone": None,
            "value": float(r.duration_s),
        })
    return pd.DataFrame(rows)
