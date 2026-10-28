"""Hall-zone dyadic occupancy and transitions — classifies each co-present
pair's distance into Hall (1966) proxemic bands and run-length-encodes
occupancy episodes per band per dyad, plus the transitions between them.

Bands (meters): intimate < 0.45, personal 0.45-1.2, social 1.2-3.6.
"Public" (> 3.6m) is not logged as its own episode stream — the pairwise
stream (proxemix/pairwise.py) is only computed up to a 3.6m search radius,
since logging every pair that is merely "not near" is combinatorially
unbounded and not proxemically meaningful; public-zone distance is the
unlogged default for any dyad not otherwise present.
"""
import numpy as np
import pandas as pd

from .groups import DEFAULT_GAP_STRIDE_FACTOR, DEFAULT_MIN_GAP_S, auto_gap_s

BANDS = [
    (0.00, 0.45, "intimate"),
    (0.45, 1.20, "personal"),
    (1.20, 3.60, "social"),
]


def _band_for(distance_m):
    for lo, hi, name in BANDS:
        if lo <= distance_m < hi:
            return name
    return None  # beyond the pairwise search radius; shouldn't occur


def detect_hall_band_episodes(pairwise_df, max_gap_s=None, debounce_frames=None,
                              min_gap_s=DEFAULT_MIN_GAP_S,
                              stride_factor=DEFAULT_GAP_STRIDE_FACTOR):
    """Returns (episodes, transitions):
    episodes — one row per (dyad, band) run: track_a, track_b, band,
      frame_start, frame_end, t_start, t_end, duration_s.
    transitions — one row per boundary between two consecutive episodes of
      the same dyad: track_a, track_b, frame, band_from, band_to.

    **The gap tolerance is in SECONDS**, derived from the sampling stride when
    left as None (`groups.auto_gap_s`). The old `debounce_frames=40` default
    was the same defect the SALSA validation found in the group filter — safe
    only at Grand Central's stride — and was repaired here 2026-10-26.
    `debounce_frames` still means frames if passed, for old callers.
    """
    df = pairwise_df.copy()
    bins = [b[0] for b in BANDS] + [BANDS[-1][1]]
    labels = [b[2] for b in BANDS]
    df["band"] = pd.cut(df["distance_m"], bins=bins, labels=labels, right=False)
    df = df.dropna(subset=["band"]).sort_values(["track_a", "track_b", "frame"])

    grp = df.groupby(["track_a", "track_b"], sort=False)
    if debounce_frames is not None:
        gap = grp["frame"].diff()
        tol = float(debounce_frames)
    else:
        gap = grp["t"].diff()
        tol = float(max_gap_s) if max_gap_s is not None else auto_gap_s(
            df, min_gap_s=min_gap_s, stride_factor=stride_factor)
    band_changed = df["band"] != grp["band"].shift()
    new_episode = gap.isna() | (gap > tol) | band_changed
    episode_id = new_episode.groupby([df["track_a"], df["track_b"]]).cumsum()
    df["episode_id"] = episode_id

    episodes = df.groupby(["track_a", "track_b", "episode_id"], as_index=False).agg(
        band=("band", "first"),
        frame_start=("frame", "min"),
        frame_end=("frame", "max"),
        t_start=("t", "min"),
        t_end=("t", "max"),
    )
    episodes["duration_s"] = episodes["t_end"] - episodes["t_start"]
    episodes = episodes.sort_values(["track_a", "track_b", "frame_start"]).reset_index(drop=True)

    transitions = []
    for (a, b), g in episodes.groupby(["track_a", "track_b"], sort=False):
        g = g.sort_values("frame_start")
        prev = None
        for _, row in g.iterrows():
            if prev is not None:
                transitions.append({
                    "track_a": a, "track_b": b, "frame": int(row.frame_start),
                    "band_from": prev, "band_to": row.band,
                })
            prev = row.band
    transitions = pd.DataFrame(transitions, columns=["track_a", "track_b", "frame", "band_from", "band_to"])
    return episodes, transitions


def episodes_to_event_rows(episodes):
    rows = []
    for _, r in episodes.iterrows():
        rows.append({
            "type": "hall_band", "actor_ids": f"{r.track_a},{r.track_b}",
            "frame_start": int(r.frame_start), "frame_end": int(r.frame_end),
            "zone": str(r.band), "value": float(r.duration_s),
        })
    return pd.DataFrame(rows)


def transitions_to_event_rows(transitions):
    rows = []
    for _, r in transitions.iterrows():
        rows.append({
            "type": "hall_transition", "actor_ids": f"{r.track_a},{r.track_b}",
            "frame_start": int(r.frame), "frame_end": int(r.frame),
            "zone": f"{r.band_from}->{r.band_to}", "value": None,
        })
    return pd.DataFrame(rows)
