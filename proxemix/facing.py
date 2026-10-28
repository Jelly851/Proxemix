"""Facing: which way each person is pointing, per track per frame.

Why this module exists
----------------------
`groups.py` used to decide conversation groups from position alone, and
`outputs/salsa_validation.md` measured what that costs: F1 0.385 against SALSA's
human F-formation labels, where published methods reach 0.83 on the same data by
also using body orientation. An F-formation is defined by where people are *and*
which way they face (Kendon 1990) — two people standing 0.7 m apart back to back
are not in conversation, and no amount of distance tuning can tell them from two
people talking.

This module supplies the missing channel, and it is explicit about the half it
cannot supply yet.

Two sources, one of them a seam
-------------------------------
1. **Motion** (built, here). While somebody walks, their direction of travel is
   their facing direction. That is a strong estimate for a transit scene like
   Grand Central, where almost everyone is moving, and a weak one in a cocktail
   party, where almost nobody is.

2. **Body keypoints** (NOT built — waits on the hackathon API key). Shoulder-line
   normal from SAM/pose keypoints is what gives a *stationary* person a facing
   direction. The seam for it is `apply_external_facing()` plus the
   `facing_source` column: any estimator that can produce
   `(track_id, frame, facing_rad, facing_conf)` fills the rows motion left
   unknown, and nothing downstream changes. There is no stub implementation
   pretending to do this.

The honest-unknown rule
-----------------------
**A stationary person's direction of motion is noise, not facing.** Below
`min_speed_m_s` the estimate is withheld: `facing_rad` is NaN, `facing_conf` is
0.0 and `facing_source` is "unknown". Consumers must branch on confidence rather
than read a number that happens to be there — `groups.py` does, falling back to
its distance-plus-transitivity path for unknown-facing people.

Confidence ramps linearly from 0 at `min_speed_m_s` to 1 at `full_speed_m_s`, so
a slow shuffle contributes weakly rather than as hard evidence.

Columns added
-------------
    facing_rad     float, radians, atan2(vy, vx) in the floor-plane frame
                   (same convention as `body_pose` in SALSA's annotations);
                   NaN where unknown.
    facing_conf    float in [0, 1]; 0.0 means unknown.
    facing_source  str, one of "motion", "unknown", or whatever an external
                   source names itself (e.g. "keypoints").

Run it: `py -m proxemix.facing` — self-test on a synthetic walker/stander pair.
"""
import numpy as np
import pandas as pd

FACING_COLUMNS = ("facing_rad", "facing_conf", "facing_source")

# Below this speed the direction of motion carries no information about facing.
# 0.25 m/s is roughly a fifth of normal walking speed (1.3 m/s, Fruin 1971) —
# fast enough that it cannot be annotation jitter, slow enough to catch a
# shuffle. Measured consequence on SALSA (3 s sampling, standing party): 2-6% of
# person-frames get a motion facing, which is the correct answer for that scene
# rather than a failure of the threshold.
DEFAULT_MIN_SPEED_M_S = 0.25
DEFAULT_FULL_SPEED_M_S = 0.60
DEFAULT_SMOOTH_S = 1.0


def _median_dt(t):
    if len(t) < 2:
        return np.nan
    d = np.diff(np.sort(np.unique(t)))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else np.nan


def add_facing(df, min_speed_m_s=DEFAULT_MIN_SPEED_M_S,
               full_speed_m_s=DEFAULT_FULL_SPEED_M_S,
               smooth_s=DEFAULT_SMOOTH_S,
               x_col="x_m", y_col="y_m", vx_col="vx_m", vy_col="vy_m",
               overwrite=False):
    """Add `facing_rad` / `facing_conf` / `facing_source` to a track table.

    Velocity is read from `vx_col`/`vy_col` when those columns exist (the
    calibration stage already produces them); otherwise it is finite-differenced
    per track on that track's own time grid. The velocity vector is then
    box-smoothed over `smooth_s` seconds *as a vector* — never as an angle,
    because averaging angles across the +/-pi wrap gives nonsense — and the
    facing angle is taken from the smoothed vector.

    `overwrite=False` leaves any existing facing columns alone, so an external
    source applied first is not clobbered by a later motion pass.
    """
    out = df.copy()
    if not overwrite and "facing_conf" in out.columns:
        return out

    has_v = vx_col in out.columns and vy_col in out.columns
    n = len(out)
    if n == 0:
        out["facing_rad"] = np.array([], dtype=float)
        out["facing_conf"] = np.array([], dtype=float)
        out["facing_source"] = np.array([], dtype=object)
        return out

    out = out.sort_values(["track_id", "frame"]).copy()

    if has_v:
        vx = out[vx_col].to_numpy(dtype=float)
        vy = out[vy_col].to_numpy(dtype=float)
    else:
        # central differences inside each track, one-sided at its ends
        vx = np.full(n, np.nan)
        vy = np.full(n, np.nan)
        tid = out["track_id"].to_numpy()
        t = out["t"].to_numpy(dtype=float)
        x = out[x_col].to_numpy(dtype=float)
        y = out[y_col].to_numpy(dtype=float)
        starts = np.flatnonzero(np.concatenate(([True], tid[1:] != tid[:-1])))
        ends = np.concatenate((starts[1:], [n]))
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            tt, xx, yy = t[s:e], x[s:e], y[s:e]
            vx[s:e] = np.gradient(xx, tt)
            vy[s:e] = np.gradient(yy, tt)

    out["_vx"] = vx
    out["_vy"] = vy

    dt = _median_dt(out["t"].to_numpy(dtype=float))
    window = 1
    if smooth_s and np.isfinite(dt) and dt > 0:
        window = int(max(1, round(smooth_s / dt)))
    if window > 1:
        g = out.groupby("track_id", sort=False)
        for c in ("_vx", "_vy"):
            out[c] = g[c].transform(
                lambda s: s.rolling(window, center=True, min_periods=1).mean()
            )

    speed = np.hypot(out["_vx"].to_numpy(dtype=float), out["_vy"].to_numpy(dtype=float))
    known = np.isfinite(speed) & (speed >= min_speed_m_s)

    facing = np.full(n, np.nan)
    facing[known] = np.arctan2(out["_vy"].to_numpy()[known], out["_vx"].to_numpy()[known])

    span = max(1e-9, full_speed_m_s - min_speed_m_s)
    conf = np.zeros(n)
    conf[known] = np.clip((speed[known] - min_speed_m_s) / span, 0.0, 1.0)

    out["facing_rad"] = facing
    out["facing_conf"] = conf
    out["facing_source"] = np.where(known, "motion", "unknown")
    out["speed_m_s"] = speed
    return out.drop(columns=["_vx", "_vy"]).reset_index(drop=True)


def apply_external_facing(df, facing_df, source="keypoints", min_conf=0.0):
    """THE SEAM for the source that is not built yet.

    `facing_df` must carry `track_id`, `frame`, `facing_rad` and optionally
    `facing_conf` (default 1.0). Rows are filled wherever the incoming
    confidence beats whatever is already there — so a pose estimator supplies
    the stationary people motion could not, and leaves the walkers alone if its
    own confidence is lower.

    This is where the SAM body-keypoint estimator plugs in once the hackathon
    API key exists: shoulder-line normal -> `facing_rad`, keypoint visibility ->
    `facing_conf`, `source="keypoints"`. Nothing else in the pipeline changes,
    and `groups.py` starts using facing for standing people the moment the
    column is populated.
    """
    if "facing_conf" not in df.columns:
        df = add_facing(df)
    out = df.copy()
    if facing_df is None or len(facing_df) == 0:
        return out

    inc = facing_df.copy()
    if "facing_conf" not in inc.columns:
        inc["facing_conf"] = 1.0
    inc = inc[inc["facing_conf"] >= min_conf]
    inc = inc[["track_id", "frame", "facing_rad", "facing_conf"]].rename(
        columns={"facing_rad": "_ext_rad", "facing_conf": "_ext_conf"})

    out = out.merge(inc, on=["track_id", "frame"], how="left")
    take = out["_ext_conf"].notna() & (
        out["_ext_conf"].fillna(-1.0) > out["facing_conf"].fillna(-1.0))
    out.loc[take, "facing_rad"] = out.loc[take, "_ext_rad"]
    out.loc[take, "facing_conf"] = out.loc[take, "_ext_conf"]
    out.loc[take, "facing_source"] = source
    return out.drop(columns=["_ext_rad", "_ext_conf"])


def facing_coverage(df):
    """How much facing the table actually has — the number to print beside any
    claim that the grouper "uses facing"."""
    if "facing_conf" not in df.columns or len(df) == 0:
        return {"n_rows": int(len(df)), "pct_known": 0.0, "by_source": {}}
    conf = df["facing_conf"].to_numpy(dtype=float)
    known = conf > 0
    out = {
        "n_rows": int(len(df)),
        "n_known": int(known.sum()),
        "pct_known": float(100.0 * known.mean()),
        "mean_conf_where_known": float(conf[known].mean()) if known.any() else 0.0,
        "by_source": {str(k): int(v) for k, v in
                      df["facing_source"].value_counts().items()},
    }
    if "speed_m_s" in df.columns:
        s = df["speed_m_s"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        out["median_speed_m_s"] = float(np.median(s)) if len(s) else None
    return out


def self_test():
    """One walker crossing east at 1.0 m/s and one person standing still, 20
    samples at 10 Hz. The walker must come out facing ~0 rad with confidence 1;
    the stander must come out unknown. Run: `py -m proxemix.facing`."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(20):
        t = i * 0.1
        rows.append({"track_id": "walker", "frame": i, "t": t,
                     "x_m": 1.0 * t, "y_m": 2.0})
        rows.append({"track_id": "stander", "frame": i, "t": t,
                     "x_m": 5.0 + rng.normal(0, 0.005),
                     "y_m": 5.0 + rng.normal(0, 0.005)})
    df = pd.DataFrame(rows)
    f = add_facing(df)
    w = f[f.track_id == "walker"]
    s = f[f.track_id == "stander"]
    results = {
        "walker_facing_deg_median": float(np.degrees(np.nanmedian(w["facing_rad"]))),
        "walker_pct_known": float(100 * (w["facing_conf"] > 0).mean()),
        "stander_pct_known": float(100 * (s["facing_conf"] > 0).mean()),
        "stander_facing_all_nan": bool(s["facing_rad"].isna().all()),
        "coverage": facing_coverage(f),
    }
    assert abs(results["walker_facing_deg_median"]) < 1.0, results
    assert results["walker_pct_known"] == 100.0, results
    assert results["stander_pct_known"] == 0.0, results
    assert results["stander_facing_all_nan"], results

    # the seam: an external source fills the stander, motion keeps the walker
    ext = pd.DataFrame([{"track_id": "stander", "frame": i,
                         "facing_rad": np.pi / 2, "facing_conf": 0.8}
                        for i in range(20)])
    merged = apply_external_facing(f, ext, source="keypoints")
    ms = merged[merged.track_id == "stander"]
    mw = merged[merged.track_id == "walker"]
    assert (ms["facing_source"] == "keypoints").all()
    assert np.allclose(ms["facing_rad"], np.pi / 2)
    assert (mw["facing_source"] == "motion").all(), "external source clobbered the walker"
    results["seam_ok"] = True
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
    print("\nfacing self-test PASSED — motion facing for the walker, honest "
          "unknown for the stander, and the external seam fills it.")
