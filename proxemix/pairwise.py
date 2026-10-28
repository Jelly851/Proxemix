"""Shared per-frame pairwise-distance computation, feeding both the
contact/proximity-episode detector and the Hall-zone dyadic band detector
(both consume the same "who was how far from whom, when" stream). Uses
scipy's cKDTree per frame so cost scales with nearby pairs, not all N^2.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compute_pairwise_distances(df, max_radius_m, x_col="x_m", y_col="y_m"):
    """df: calibrated track table (track_id, frame, t, x_m, y_m, ...).
    Returns a long DataFrame: frame, t, track_a, track_b, distance_m — one
    row per co-present pair within max_radius_m at that frame. track_a <
    track_b by string order (each unordered pair appears once per frame).
    """
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        if len(g) < 2:
            continue
        ids = g["track_id"].to_numpy()
        xy = g[[x_col, y_col]].to_numpy()
        tree = cKDTree(xy)
        pairs = tree.query_pairs(r=max_radius_m, output_type="ndarray")
        if len(pairs) == 0:
            continue
        d = np.hypot(
            xy[pairs[:, 0], 0] - xy[pairs[:, 1], 0],
            xy[pairs[:, 0], 1] - xy[pairs[:, 1], 1],
        )
        a_ids = ids[pairs[:, 0]]
        b_ids = ids[pairs[:, 1]]
        lo = np.minimum(a_ids, b_ids)
        hi = np.maximum(a_ids, b_ids)
        t_val = g["t"].iloc[0]
        for i in range(len(pairs)):
            rows.append((frame, t_val, lo[i], hi[i], d[i]))
    return pd.DataFrame(rows, columns=["frame", "t", "track_a", "track_b", "distance_m"])
