"""Learned pairwise grouper — replaces the hand-set geometric thresholds in
`groups.py` with a classifier trained on SALSA's own labelled pairs.

WHAT THIS IS
-------------
For every pair of people present in a frame: predict "are these two in a
conversation group together?" from five hand-made feature families —
distance, heading difference, co-movement, local crowding, and facing
alignment wherever motion supplies it (`proxemix.facing`) — then build groups
by graph clustering over the predicted *probabilities*, not from the raw
pairwise predictions. Labels come free from SALSA's `fformationGT.csv`: a
pair is positive at a timestamp if both members are listed together in any
ground-truth group there.

Two models: `HistGradientBoostingClassifier` (main — tolerates missing
features natively, which matters because facing is unknown for 96-97% of
SALSA person-frames) and `LogisticRegression` (baseline, with median
imputation + a missing-value indicator per feature, since it cannot take
NaN directly).

NO LEAKAGE, EVER. Every number this module or `salsa_validation.py --method
learned` reports is scored on the SALSA part the model did *not* train on.
`cross_part_report()` is the one function that does both directions and is
the only entry point `salsa_validation.py` calls for the headline numbers.

GROUPING FROM PROBABILITIES — divisive, not single-linkage
------------------------------------------------------------
Thresholding a probability graph and taking connected components is exactly
DBSCAN's single-linkage mistake in a learned disguise: one confident pair at
each end of a six-person chain welds the room into one group. So clustering
here mirrors `groups.py`'s divisive design, with probability standing in for
geometry: a connected component (edges where P >= tau_edge) is accepted
whole only if *every* pair inside it clears a stricter floor
(min pairwise P >= tau_group); a component that fails is split along the
weakest seam of its own probability graph (spectral bisection — the same
normalised-cut idea `groups.py` uses, reimplemented here rather than
imported, so this module does not depend on `groups.py`'s private API) and
each half is reconsidered. People who end up alone are noise.

Both thresholds are chosen on the TRAINING part only (a small sweep against
that part's own ground truth), never on the test part — see
`fit_thresholds()`. That carries a small optimism bias of its own, disclosed
in `outputs/salsa_validation.md` section 19.3 rather than hidden.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proxemix.groups import CLUSTER_COLUMNS

# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

# Every feature is symmetric under swapping the pair's two members, so the
# unordered pair (i, j) and (j, i) always produce the identical row — no
# ordering convention has to be invented or remembered downstream.
FEATURE_NAMES = [
    "dist_m",             # distance — straight-line separation
    "heading_diff_rad",   # heading difference — |facing_i - facing_j|, wrapped; NaN unless both known
    "facing_known_both",  # 1.0 if both members have a motion facing, else 0.0
    "facing_align_mean",  # facing — mean of how much each known-facing member points at the other
    "facing_align_min",   # facing — the worse of the two alignments (NaN if neither known)
    "co_move_cos",        # co-movement — cosine angle between velocity vectors; NaN if either near-still
    "speed_sum_m_s",      # co-movement — combined speed, context for co_move_cos
    "speed_diff_m_s",     # co-movement — how differently fast the two are moving
    "local_crowding",     # local crowding — others within 2 m of the pair's midpoint
    "n_people_frame",     # local crowding — context: how busy the whole frame is
]

CROWD_RADIUS_M = 2.0
MIN_SPEED_FOR_COMOVE_M_S = 0.15  # looser than facing.py's 0.25 m/s facing cutoff —
# co-movement only needs "is this person moving", not "is this a trustworthy facing"


def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _velocity(frame_df, smooth_window=1):
    """Per-track velocity inside one frame's worth of rows is meaningless on
    its own — SALSA samples every 3 s — so velocity is computed once per
    *track* across the whole part before frames are split out. See
    `_add_velocity_whole_track`, which this is a thin per-row reader for."""
    return frame_df["_vx"].to_numpy(dtype=float), frame_df["_vy"].to_numpy(dtype=float)


def _add_velocity_whole_track(tracks, smooth_s=3.0):
    """Finite-difference velocity per track over the FULL track (not one
    frame), box-smoothed as a vector over `smooth_s`. Same method as
    `proxemix.facing.add_facing`'s own velocity pass, reimplemented here
    because that module drops its `_vx`/`_vy` columns once it has used them
    and this module needs the raw vectors for co-movement, independent of
    whether a frame's speed clears facing.py's stricter facing-confidence
    threshold."""
    out = tracks.sort_values(["track_id", "frame"]).copy()
    n = len(out)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    tid = out["track_id"].to_numpy()
    t = out["t"].to_numpy(dtype=float)
    x = out["x_m"].to_numpy(dtype=float)
    y = out["y_m"].to_numpy(dtype=float)
    starts = np.flatnonzero(np.concatenate(([True], tid[1:] != tid[:-1])))
    ends = np.concatenate((starts[1:], [n]))
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        tt, xx, yy = t[s:e], x[s:e], y[s:e]
        vx[s:e] = np.gradient(xx, tt)
        vy[s:e] = np.gradient(yy, tt)
    out["_vx"], out["_vy"] = vx, vy
    dt = np.median(np.diff(np.unique(t))) if len(np.unique(t)) > 1 else np.nan
    window = 1
    if smooth_s and np.isfinite(dt) and dt > 0:
        window = int(max(1, round(smooth_s / dt)))
    if window > 1:
        g = out.groupby("track_id", sort=False)
        for c in ("_vx", "_vy"):
            out[c] = g[c].transform(
                lambda s: s.rolling(window, center=True, min_periods=1).mean()
            )
    return out


def prepare_tracks(tracks):
    """Attach the raw velocity columns (`_vx`, `_vy`) this module's features
    need, on top of whatever facing columns (`facing_rad`/`facing_conf`) the
    caller has already applied via `proxemix.facing`. Call once per part."""
    return _add_velocity_whole_track(tracks)


def _pair_features(sub):
    """sub: one frame's rows (prepared via `prepare_tracks`), >= 2 of them.
    Returns (ids, feat) where feat is an (n_pairs, len(FEATURE_NAMES)) array
    in the order of `itertools.combinations(range(n), 2)`, and ids is the
    frame's track_id array (so pair k corresponds to ids[i], ids[j])."""
    ids = sub["track_id"].to_numpy()
    xy = sub[["x_m", "y_m"]].to_numpy(dtype=float)
    vx = sub["_vx"].to_numpy(dtype=float)
    vy = sub["_vy"].to_numpy(dtype=float)
    speed = np.hypot(vx, vy)
    has_facing = "facing_rad" in sub.columns
    facing = sub["facing_rad"].to_numpy(dtype=float) if has_facing else np.full(len(sub), np.nan)
    fconf = sub["facing_conf"].to_numpy(dtype=float) if has_facing else np.zeros(len(sub))
    known = np.isfinite(facing) & (fconf > 0)

    n = len(sub)
    i_idx, j_idx = np.triu_indices(n, 1)

    dx = xy[j_idx, 0] - xy[i_idx, 0]
    dy = xy[j_idx, 1] - xy[i_idx, 1]
    dist = np.hypot(dx, dy)
    bearing_ij = np.arctan2(dy, dx)        # i -> j
    bearing_ji = _wrap_pi(bearing_ij + np.pi)  # j -> i

    ki, kj = known[i_idx], known[j_idx]
    both_known = ki & kj
    heading_diff = np.full(len(i_idx), np.nan)
    heading_diff[both_known] = np.abs(_wrap_pi(facing[i_idx][both_known] - facing[j_idx][both_known]))

    align_i = np.full(len(i_idx), np.nan)
    align_j = np.full(len(i_idx), np.nan)
    align_i[ki] = np.cos(_wrap_pi(facing[i_idx][ki] - bearing_ij[ki]))
    align_j[kj] = np.cos(_wrap_pi(facing[j_idx][kj] - bearing_ji[kj]))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        align_stack = np.vstack([align_i, align_j])
        align_mean = np.nanmean(align_stack, axis=0)
        align_min = np.nanmin(align_stack, axis=0)
    neither = ~ki & ~kj
    align_mean[neither] = np.nan
    align_min[neither] = np.nan

    si, sj = speed[i_idx], speed[j_idx]
    moving = np.isfinite(si) & np.isfinite(sj) & (si >= MIN_SPEED_FOR_COMOVE_M_S) & (sj >= MIN_SPEED_FOR_COMOVE_M_S)
    co_move = np.full(len(i_idx), np.nan)
    denom = (si * sj)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_v = (vx[i_idx] * vx[j_idx] + vy[i_idx] * vy[j_idx]) / np.where(denom > 1e-9, denom, np.nan)
    co_move[moving] = np.clip(cos_v[moving], -1.0, 1.0)
    speed_sum = np.where(np.isfinite(si) & np.isfinite(sj), np.nan_to_num(si) + np.nan_to_num(sj), np.nan)
    speed_diff = np.where(np.isfinite(si) & np.isfinite(sj), np.abs(np.nan_to_num(si) - np.nan_to_num(sj)), np.nan)

    mid = (xy[i_idx] + xy[j_idx]) / 2.0
    all_d_to_mid = np.hypot(xy[:, 0][None, :] - mid[:, 0:1], xy[:, 1][None, :] - mid[:, 1:2])
    within = all_d_to_mid <= CROWD_RADIUS_M
    pair_mask = np.zeros((len(i_idx), n), dtype=bool)
    pair_mask[np.arange(len(i_idx)), i_idx] = True
    pair_mask[np.arange(len(i_idx)), j_idx] = True
    crowding = (within & ~pair_mask).sum(axis=1).astype(float)

    feat = np.column_stack([
        dist, heading_diff, both_known.astype(float), align_mean, align_min,
        co_move, speed_sum, speed_diff, crowding, np.full(len(i_idx), float(n)),
    ])
    return ids, i_idx, j_idx, feat


def build_pair_dataset(tracks, gt):
    """tracks: prepared via `prepare_tracks`, carrying facing columns if any
    facing mode is in use. gt: `salsa_adapter.load_salsa_fformations` output.

    Returns a DataFrame with one row per (frame, unordered pair of people
    present in that frame): frame, id_i, id_j, label, then the FEATURE_NAMES
    columns. Only frames SALSA annotated groups for are used, matching
    `salsa_validation.run_part`'s own evaluation frames exactly.
    """
    frames = sorted(gt["frame"].unique().tolist())
    gt_members = {}
    for frame, g in gt.groupby("frame"):
        gt_members[int(frame)] = [frozenset(k.split(",")) for k in g["group_key"]]

    rows_ids, rows_feat, rows_label, rows_frame = [], [], [], []
    for frame, sub in tracks[tracks["frame"].isin(frames)].groupby("frame"):
        if len(sub) < 2:
            continue
        ids, i_idx, j_idx, feat = _pair_features(sub)
        groups_here = gt_members.get(int(frame), [])
        label = np.zeros(len(i_idx), dtype=int)
        for gi, (ii, jj) in enumerate(zip(i_idx, j_idx)):
            a, b = ids[ii], ids[jj]
            for grp in groups_here:
                if a in grp and b in grp:
                    label[gi] = 1
                    break
        rows_ids.append(np.column_stack([ids[i_idx], ids[j_idx]]))
        rows_feat.append(feat)
        rows_label.append(label)
        rows_frame.append(np.full(len(i_idx), int(frame)))

    id_pairs = np.concatenate(rows_ids, axis=0)
    feat = np.concatenate(rows_feat, axis=0)
    label = np.concatenate(rows_label, axis=0)
    frame_col = np.concatenate(rows_frame, axis=0)

    df = pd.DataFrame(feat, columns=FEATURE_NAMES)
    df.insert(0, "frame", frame_col)
    df.insert(1, "id_i", id_pairs[:, 0])
    df.insert(2, "id_j", id_pairs[:, 1])
    df["label"] = label
    return df


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def make_model(kind):
    """kind: 'hgb' (main model) or 'logreg' (baseline). Both use
    `class_weight="balanced"` — most pairs in any frame are not a group, and
    unweighted fitting under-predicts the minority (together) class."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=4,
            l2_regularization=1.0, class_weight="balanced", random_state=0,
        )
    if kind == "logreg":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    raise ValueError(f"unknown model kind {kind!r} — use 'hgb' or 'logreg'")


def fit_model(pair_df, kind):
    model = make_model(kind)
    X = pair_df[FEATURE_NAMES].to_numpy(dtype=float)
    y = pair_df["label"].to_numpy(dtype=int)
    model.fit(X, y)
    return model


def fit_thresholds(model, train_pair_df, train_tracks,
                    tau_edge_grid=(0.15, 0.2, 0.3, 0.4, 0.5, 0.6),
                    tau_group_grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Pick (tau_edge, tau_group) by sweeping a 2-D grid on the TRAINING
    part's own ground truth only — never the test part. `tau_edge` governs
    which pairs may even be linked; `tau_group` (always >= tau_edge) is the
    stricter floor every pair inside an accepted group must clear — the
    learned analogue of `groups.py`'s separate linking-radius and
    transitivity-guard thresholds. Returns (tau_edge, tau_group, sweep_table)
    where sweep_table is a list of dicts for the write-up."""
    from proxemix.salsa_validation import _groups_by_frame, evaluate

    gt_by_frame = {}
    for frame, g in train_pair_df.groupby("frame"):
        groups_here = []
        # reconstruct ground-truth member sets for this frame from the label
        # column via a union-find over positive pairs, so this sweep never
        # touches `fformationGT.csv` a second time with different code.
        pos = g[g["label"] == 1]
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                x = parent[x]
            return x

        for _, r in pos.iterrows():
            a, b = find(r.id_i), find(r.id_j)
            parent.setdefault(r.id_i, r.id_i)
            parent.setdefault(r.id_j, r.id_j)
            if a != b:
                parent[a] = b
        comp = {}
        for node in parent:
            comp.setdefault(find(node), set()).add(node)
        groups_here = [frozenset(m) for m in comp.values() if len(m) >= 2]
        gt_by_frame[int(frame)] = groups_here

    cache = predict_all_frames(train_tracks, model)
    frames = sorted(gt_by_frame)
    results = []
    best = (0.5, 0.5, -1.0)
    for te in tau_edge_grid:
        for tg in tau_group_grid:
            if tg < te:
                continue
            clusters = cluster_from_cache(cache, te, tg)
            det_by_frame = _groups_by_frame(clusters) if len(clusters) else {}
            res = evaluate(gt_by_frame, det_by_frame, 2 / 3, frames=frames)
            results.append({"tau_edge": te, "tau_group": tg, "f1": res["f1"],
                            "precision": res["precision"], "recall": res["recall"]})
            if res["f1"] > best[2]:
                best = (te, tg, res["f1"])
    return best[0], best[1], results


# ---------------------------------------------------------------------------
# clustering from predicted probabilities
# ---------------------------------------------------------------------------

def _connected_components(edge):
    n = len(edge)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in np.flatnonzero(edge[v]):
                w = int(w)
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(comp)
    return comps


def _spectral_bisect(idx, affinity):
    """Split `idx` (a list of global indices) in two along the weakest seam
    of the sub-graph `affinity[idx, idx]` induces — the Fiedler vector of the
    normalised Laplacian, same normalised-min-cut idea `groups.py` uses for
    geometric affinity, applied here to learned pairwise probabilities."""
    n = len(idx)
    if n <= 2:
        return [idx[0]], idx[1:]
    A = affinity[np.ix_(idx, idx)].astype(float).copy()
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    L = np.eye(n) - A * dinv[:, None] * dinv[None, :]
    try:
        _, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:  # pragma: no cover
        return [idx[0]], idx[1:]
    f = vecs[:, 1] * dinv
    order = np.argsort(f, kind="stable")
    f_sorted = f[order]
    gaps = np.diff(f_sorted)
    cut = int(np.argmax(gaps)) + 1 if len(gaps) else 1
    left = [idx[i] for i in order[:cut]]
    right = [idx[i] for i in order[cut:]]
    if not left or not right:
        return [idx[0]], idx[1:]
    return left, right


def _divide(idx, prob, tau_edge, tau_group, out, depth=0):
    if len(idx) < 2 or depth > 20:
        return
    sub = prob[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    if sub[iu].min() >= tau_group:
        out.append(list(idx))
        return
    left, right = _spectral_bisect(idx, prob)
    for half in (left, right):
        if len(half) < 2:
            continue
        sub_edge = prob[np.ix_(half, half)] >= tau_edge
        np.fill_diagonal(sub_edge, False)
        for comp in _connected_components(sub_edge):
            comp_idx = [half[c] for c in comp]
            if len(comp_idx) >= 2:
                _divide(comp_idx, prob, tau_edge, tau_group, out, depth + 1)


def cluster_frame_probs(prob, tau_edge=0.5, tau_group=0.5):
    """prob: symmetric (n, n) probability matrix, diagonal ignored. Returns a
    list of lists of local indices (0..n-1), the divisive build described in
    the module docstring."""
    n = len(prob)
    edge = prob >= tau_edge
    np.fill_diagonal(edge, False)
    out = []
    for comp in _connected_components(edge):
        if len(comp) >= 2:
            _divide(comp, prob, tau_edge, tau_group, out)
    return out


def predict_frame_prob_matrix(model, sub):
    """sub: one frame's prepared rows. Returns (ids, prob) — ids the
    track_id array, prob the symmetric (n, n) predicted-probability matrix."""
    ids, i_idx, j_idx, feat = _pair_features(sub)
    n = len(sub)
    prob_vec = model.predict_proba(feat)[:, 1]
    prob = np.zeros((n, n))
    prob[i_idx, j_idx] = prob_vec
    prob[j_idx, i_idx] = prob_vec
    return ids, prob


def predict_all_frames(tracks, model):
    """Run the classifier once per frame and cache (ids, prob, t) per frame,
    so a threshold sweep (`fit_thresholds`) or a re-cluster at a different
    (tau_edge, tau_group) never re-runs inference."""
    cache = {}
    for frame, sub in tracks.groupby("frame", sort=True):
        if len(sub) < 2:
            continue
        ids, prob = predict_frame_prob_matrix(model, sub)
        cache[int(frame)] = (ids, prob, sub["t"].iloc[0])
    return cache


def cluster_from_cache(cache, tau_edge, tau_group, min_samples=2):
    rows = []
    for frame, (ids, prob, t_val) in cache.items():
        for local in cluster_frame_probs(prob, tau_edge=tau_edge, tau_group=tau_group):
            if len(local) < min_samples:
                continue
            rows.append({
                "frame": frame, "t": t_val,
                "group_key": ",".join(sorted(ids[local].tolist())),
                "n_members": len(local),
            })
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)


def detect_frame_clusters_learned(tracks, model, tau_edge=0.5, tau_group=0.5, min_samples=2):
    """Same output contract as `groups.detect_frame_clusters`: a DataFrame
    with columns frame, t, group_key, n_members — one row per detected group
    per frame — so it is a drop-in alternative detector for
    `salsa_validation.py`. `tracks` must already carry prepared velocity
    (`prepare_tracks`) and, optionally, facing columns."""
    cache = predict_all_frames(tracks, model)
    return cluster_from_cache(cache, tau_edge, tau_group, min_samples)


# ---------------------------------------------------------------------------
# cross-part orchestration — the one entry point salsa_validation.py calls
# ---------------------------------------------------------------------------

def feature_importances(model, kind, pair_df):
    """Permutation importance of each feature, measured on `pair_df` (meant
    to be the held-out test part, never the part the model trained on) using
    ROC-AUC as the scoring metric. Returns a list of (name, mean_importance,
    std) sorted by importance, descending."""
    X = pair_df[FEATURE_NAMES].to_numpy(dtype=float)
    y = pair_df["label"].to_numpy(dtype=int)
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=8,
                                random_state=0, n_jobs=1)
    order = np.argsort(r.importances_mean)[::-1]
    return [(FEATURE_NAMES[i], float(r.importances_mean[i]), float(r.importances_std[i]))
            for i in order]


def cross_part_direction(train_part_tracks, train_part_gt, test_part_tracks, test_part_gt, kind):
    """Train on one part, score on the other. Returns a dict with the fitted
    model, the chosen thresholds, the test-part pair dataset (for importance
    reporting) and the detected clusters DataFrame for `salsa_validation.py`
    to evaluate with its own `evaluate()`/`pool()`."""
    train_pairs = build_pair_dataset(train_part_tracks, train_part_gt)
    model = fit_model(train_pairs, kind)
    tau_edge, tau_group, sweep = fit_thresholds(model, train_pairs, train_part_tracks)
    clusters = detect_frame_clusters_learned(test_part_tracks, model, tau_edge=tau_edge, tau_group=tau_group)
    test_pairs = build_pair_dataset(test_part_tracks, test_part_gt)
    return {
        "model": model, "tau_edge": tau_edge, "tau_group": tau_group,
        "threshold_sweep": sweep, "train_pairs": train_pairs, "test_pairs": test_pairs,
        "clusters": clusters, "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
    }


def self_test():
    """A hand-built two-frame, three-person dataset: A and B always walk
    side by side 0.8 m apart facing the same way (a real pair), C stands 3 m
    away alone. Checks the feature builder and the divisive clustering both
    separate C from A/B and keep A+B together. Run: `py -m
    proxemix.learned_groups`."""
    rows = []
    for f in range(6):
        t = f * 1.0
        rows.append({"track_id": "A", "frame": f, "t": t, "x_m": 0.5 * f, "y_m": 0.0})
        rows.append({"track_id": "B", "frame": f, "t": t, "x_m": 0.5 * f, "y_m": 0.8})
        rows.append({"track_id": "C", "frame": f, "t": t, "x_m": 5.0, "y_m": 5.0})
    tracks = prepare_tracks(pd.DataFrame(rows))
    sub = tracks[tracks["frame"] == 3]
    ids, i_idx, j_idx, feat = _pair_features(sub)
    pairs = list(zip(ids[i_idx], ids[j_idx]))
    ab = pairs.index(("A", "B")) if ("A", "B") in pairs else pairs.index(("B", "A"))
    ac = pairs.index(("A", "C")) if ("A", "C") in pairs else pairs.index(("C", "A"))
    assert feat[ab][FEATURE_NAMES.index("dist_m")] < feat[ac][FEATURE_NAMES.index("dist_m")]
    assert feat[ab][FEATURE_NAMES.index("co_move_cos")] > 0.9, feat[ab]

    n = len(sub)
    prob = np.full((n, n), 0.05)
    a, b, c = (int(np.flatnonzero(ids == x)[0]) for x in ("A", "B", "C"))
    prob[a, b] = prob[b, a] = 0.9
    groups = cluster_frame_probs(prob, tau_edge=0.5, tau_group=0.5)
    member_sets = [frozenset(ids[g]) for g in groups]
    assert frozenset({"A", "B"}) in member_sets, member_sets
    assert not any("C" in m for m in member_sets), member_sets
    return {"pairs_checked": len(pairs), "ab_dist_m": float(feat[ab][0]),
            "ab_co_move_cos": float(feat[ab][5]), "groups_found": [sorted(m) for m in member_sets]}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2, default=str))
    print("\nlearned_groups self-test PASSED — features are symmetric under pair "
          "order, the side-by-side co-moving pair scores high co-movement, and "
          "divisive clustering over a toy probability matrix keeps A+B together "
          "while refusing C.")
