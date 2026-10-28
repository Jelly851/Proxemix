"""Within-clip outlier triage: ranks tracks by statistical strangeness
relative to the rest of the dataset, with the reason attached — leads for
a human reviewer, never accusations. Not an "event" (no frame range makes
sense for a whole-track property), so this lands as its own summary table
rather than `events` rows.
"""
import numpy as np
import pandas as pd

FEATURES = ["duration_s", "distance_m", "mean_speed_m_s", "max_speed_m_s", "directness_ratio"]


def build_track_features(calibrated_df, od_table):
    rows = []
    for track_id, g in calibrated_df.sort_values("frame").groupby("track_id", sort=False):
        speed = np.hypot(g["vx_m"].to_numpy(), g["vy_m"].to_numpy())
        duration = float(g["t"].iloc[-1] - g["t"].iloc[0])
        xy = g[["x_m", "y_m"]].to_numpy()
        d = np.diff(xy, axis=0) if len(xy) >= 2 else np.zeros((0, 2))
        distance = float(np.hypot(d[:, 0], d[:, 1]).sum())
        rows.append({
            "track_id": track_id,
            "duration_s": duration,
            "distance_m": distance,
            "mean_speed_m_s": distance / duration if duration > 0 else 0.0,
            "max_speed_m_s": float(speed.max()) if len(speed) else 0.0,
            "n_points": len(g),
        })
    feats = pd.DataFrame(rows)
    feats = feats.merge(od_table[["track_id", "directness_ratio"]], on="track_id", how="left")
    return feats


def _robust_z(series):
    med = series.median()
    mad = (series - med).abs().median()
    scale = mad * 1.4826
    if scale == 0 or np.isnan(scale):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - med) / scale


def rank_outliers(feats, top_n=25):
    z = pd.DataFrame({f: _robust_z(feats[f].fillna(feats[f].median())) for f in FEATURES})
    strangeness = z.abs().max(axis=1)
    worst_feature = z.abs().idxmax(axis=1)

    out = feats.copy()
    out["strangeness_score"] = strangeness.to_numpy()
    out["worst_feature"] = worst_feature.to_numpy()
    out["worst_feature_z"] = [z.loc[i, worst_feature.loc[i]] for i in z.index]
    out["reason"] = [
        f"{row.worst_feature}={getattr(row, row.worst_feature):.2f}, "
        f"z={row.worst_feature_z:.1f} vs the dataset's median track"
        for row in out.itertuples()
    ]
    return out.sort_values("strangeness_score", ascending=False).head(top_n).reset_index(drop=True)


def rank_to_event_rows(ranked):
    rows = []
    for _, r in ranked.iterrows():
        rows.append({
            "type": "outlier", "actor_ids": r.track_id,
            "frame_start": None, "frame_end": None,
            "zone": None, "value": float(r.strangeness_score),
        })
    return pd.DataFrame(rows)
