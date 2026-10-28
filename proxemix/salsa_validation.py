"""SALSA validation — scores Proxemix's conversation-group detection against
SALSA's human F-formation annotations, and prints the precision/recall/F1 that
`outputs/salsa_validation.md` reports.

Run it:
    py -m proxemix.salsa_validation                  # default parameters, both parts
    py -m proxemix.salsa_validation --sweep          # also sweep eps_m (tuned score)
    py -m proxemix.salsa_validation --json out.json  # dump every number

Nothing in `proxemix/groups.py` is modified. `detect_frame_clusters` is called
with its own default arguments for the headline number; where a non-default
argument is used it is passed explicitly and labelled in the output.


THE MEASURE — group-detection F1 under the tolerant-match criterion
-------------------------------------------------------------------
Implemented from Setti, Russell, Bassetti & Cristani, "F-Formation Detection:
Individuating Free-Standing Conversational Groups in Images", PLOS ONE 10(5):
e0123783, 2015 (the GCFF paper), which is the convention the F-formation
literature reports against. Quoted from it verbatim:

    "We consider a group as correctly estimated if at least ceil(T * |G|) of
    their members are found by the grouping method and correctly detected by
    the tracker, and if no more than 1 - ceil(T * |G|) false subjects (of the
    detected tracks) are identified"

    "precision = TP/(TP+FP), recall = TP/(TP+FN)"
    "F1 = 2 * (precision * recall) / (precision + recall)"

ONE CORRECTION TO THAT QUOTE, made deliberately and visibly. The false-subject
bound as typeset — "1 - ceil(T * |G|)" — is degenerate: for |G| = 3 and
T = 2/3 it allows at most 1 - 2 = -1 false subjects, which no detection can
satisfy. The bound the field actually uses, and which the same criterion is
stated with in the Hung & Kroese lineage this measure comes from ("Detecting
F-formations as dominant sets", ICMI 2011), is

    no more than (1 - T) * |G| false subjects

so that is what `tolerant_match` implements, with the count floored to an
integer. Sanity of the result: at T = 2/3 a ground-truth dyad (|G| = 2) needs
ceil(4/3) = 2 members found and floor(2/3) = 0 false subjects — i.e. dyads must
be matched exactly, which is the known behaviour of the 2/3 criterion. At T = 1
every group must be matched exactly. Both are reported.

Three conventions the paper leaves open, each decided here and stated in
`outputs/salsa_validation.md` section 4:

* Singletons are excluded from ground truth and detections alike. An
  F-formation involves two or more people; the paper calls a lone individual a
  "spurious F-formation" kept "for the sake of mathematical simplicity", and
  its per-cardinality tables run k = 2..6. `--keep-singletons` reports the
  other way as a sensitivity check.
* Matching is one-to-one within a frame, solved as maximum-cardinality
  bipartite matching over the criterion-satisfying pairs, so one large
  detection cannot be credited against several true groups.
* TP/FP/FN are pooled over all evaluated frames and divided once
  (micro-average). The macro-average over frames is also printed.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from proxemix import facing as facing_mod
from proxemix import groups as groups_mod
from proxemix import learned_groups as learned_mod
from proxemix.groups import detect_frame_clusters, detect_group_episodes
from proxemix.salsa_adapter import (
    PART_LABELS,
    SALSA_FPS,
    SALSA_PARTS,
    load_salsa_fformations,
    load_salsa_tracks,
)

DEFAULT_SALSA_ROOT = os.path.join("data", "salsa")

# proxemix.groups' own signature defaults, read from the module rather than
# retyped, so the report can print "these were the defaults" without guessing
# and cannot drift from them.
GROUPS_DEFAULT_EPS_M = groups_mod.DEFAULT_EPS_M
GROUPS_DEFAULT_MIN_SAMPLES = 2
GROUPS_DEFAULT_METHOD = groups_mod.DEFAULT_METHOD
GROUPS_DEFAULT_BUILD = groups_mod.DEFAULT_BUILD

# The persistence filter now counts seconds and derives its gap tolerance from
# the data's own sampling stride (groups.auto_gap_s), so nothing dataset-specific
# has to be passed here any more. SALSA's 3 s stride yields a 7.5 s tolerance —
# one dropped sample, with slack. The old `debounce_frames=90` workaround for
# this part of the harness is gone; the defect it worked around is fixed in
# groups.py.
SALSA_MIN_DURATION_S = groups_mod.DEFAULT_MIN_DURATION_S

# Facing sources the harness can feed the grouper:
#   "motion" — proxemix.facing.add_facing, the shipped keyless estimator. On
#              SALSA's 3 s standing-party sampling this leaves most people
#              unknown, which is the honest answer for that scene.
#   "none"   — no facing columns at all; the grouper's distance-plus-
#              transitivity fallback only.
#   "oracle" — SALSA's own annotated Body_Pose as facing, confidence 1.0. NOT a
#              Proxemix capability: it is the upper bound the SAM-keypoint source
#              in facing.py is aiming at, and it is what makes the published
#              position+orientation number (F1 0.83) comparable.
FACING_MODES = ("motion", "none", "oracle")


def _members(group_key):
    return frozenset(group_key.split(","))


def tolerant_match(gt, det, T):
    """True if detected member set `det` correctly estimates ground-truth set
    `gt` at tolerance threshold T. See the module docstring for the source and
    the one correction made to it."""
    gt, det = set(gt), set(det)
    common = len(gt & det)
    needed = math.ceil(T * len(gt))
    false_allowed = math.floor((1.0 - T) * len(gt))
    return common >= needed and len(det - gt) <= false_allowed


def score_frame(gt_groups, det_groups, T):
    """One frame. gt_groups/det_groups: lists of member sets.
    Returns (tp, fp, fn, matches) where matches is a list of (gt_idx, det_idx).
    Maximum-cardinality one-to-one matching over criterion-satisfying pairs."""
    if not gt_groups or not det_groups:
        return 0, len(det_groups), len(gt_groups), []

    ok = np.zeros((len(gt_groups), len(det_groups)), dtype=bool)
    for i, g in enumerate(gt_groups):
        for j, d in enumerate(det_groups):
            ok[i, j] = tolerant_match(g, d, T)

    # cost 0 for an allowed pair, 1 for a disallowed one; minimising the total
    # over a square-padded matrix maximises the number of allowed pairs used.
    n = max(len(gt_groups), len(det_groups))
    cost = np.ones((n, n))
    cost[: len(gt_groups), : len(det_groups)] = np.where(ok, 0.0, 1.0)
    rows, cols = linear_sum_assignment(cost)

    matches = [
        (int(i), int(j))
        for i, j in zip(rows, cols)
        if i < len(gt_groups) and j < len(det_groups) and ok[i, j]
    ]
    tp = len(matches)
    return tp, len(det_groups) - tp, len(gt_groups) - tp, matches


def _groups_by_frame(df):
    """frame -> list of member sets, from any table carrying frame/group_key."""
    out = {}
    for frame, g in df.groupby("frame", sort=True):
        out[int(frame)] = [_members(k) for k in g["group_key"]]
    return out


def persistent_clusters(clusters, min_duration_s=SALSA_MIN_DURATION_S,
                        max_gap_s=None):
    """Keep only the (frame, group_key) rows that fall inside an episode which
    survived proxemix.groups' persistence filter — i.e. what Proxemix actually
    *reports* as a conversation group, rather than every momentary cluster."""
    episodes = detect_group_episodes(
        clusters, max_gap_s=max_gap_s, min_duration_s=min_duration_s
    )
    if episodes.empty:
        return clusters.iloc[0:0].copy()
    keep = np.zeros(len(clusters), dtype=bool)
    key = clusters["group_key"].to_numpy()
    frame = clusters["frame"].to_numpy()
    for _, e in episodes.iterrows():
        keep |= (key == e.group_key) & (frame >= e.frame_start) & (frame <= e.frame_end)
    return clusters[keep].copy()


def evaluate(gt_by_frame, det_by_frame, T, frames=None):
    """Pooled and macro precision/recall/F1 plus per-cardinality breakdowns and
    the raw per-frame records needed for failure analysis."""
    if frames is None:
        frames = sorted(set(gt_by_frame) | set(det_by_frame))

    tp = fp = fn = 0
    per_frame = []
    gt_card_tp, gt_card_total = {}, {}
    det_card_tp, det_card_total = {}, {}
    unmatched_gt, unmatched_det = [], []

    for f in frames:
        g = gt_by_frame.get(f, [])
        d = det_by_frame.get(f, [])
        ftp, ffp, ffn, matches = score_frame(g, d, T)
        tp += ftp
        fp += ffp
        fn += ffn

        matched_g = {i for i, _ in matches}
        matched_d = {j for _, j in matches}
        for i, grp in enumerate(g):
            k = len(grp)
            gt_card_total[k] = gt_card_total.get(k, 0) + 1
            if i in matched_g:
                gt_card_tp[k] = gt_card_tp.get(k, 0) + 1
            else:
                unmatched_gt.append({"frame": f, "size": k,
                                     "members": ",".join(sorted(grp))})
        for j, grp in enumerate(d):
            k = len(grp)
            det_card_total[k] = det_card_total.get(k, 0) + 1
            if j in matched_d:
                det_card_tp[k] = det_card_tp.get(k, 0) + 1
            else:
                unmatched_det.append({"frame": f, "size": k,
                                      "members": ",".join(sorted(grp))})

        p = ftp / (ftp + ffp) if (ftp + ffp) else (1.0 if not ffn else 0.0)
        r = ftp / (ftp + ffn) if (ftp + ffn) else (1.0 if not ffp else 0.0)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_frame.append({"frame": f, "tp": ftp, "fp": ffp, "fn": ffn,
                          "precision": p, "recall": r, "f1": f1,
                          "n_gt": len(g), "n_det": len(d)})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pf = pd.DataFrame(per_frame)

    return {
        "T": T,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "macro_precision": float(pf["precision"].mean()) if len(pf) else 0.0,
        "macro_recall": float(pf["recall"].mean()) if len(pf) else 0.0,
        "macro_f1": float(pf["f1"].mean()) if len(pf) else 0.0,
        "n_frames": len(pf),
        "recall_by_gt_size": {
            k: {"n": gt_card_total[k], "tp": gt_card_tp.get(k, 0),
                "recall": gt_card_tp.get(k, 0) / gt_card_total[k]}
            for k in sorted(gt_card_total)
        },
        "precision_by_det_size": {
            k: {"n": det_card_total[k], "tp": det_card_tp.get(k, 0),
                "precision": det_card_tp.get(k, 0) / det_card_total[k]}
            for k in sorted(det_card_total)
        },
        "_per_frame": pf,
        "_unmatched_gt": pd.DataFrame(unmatched_gt),
        "_unmatched_det": pd.DataFrame(unmatched_det),
    }


def failure_analysis(gt_by_frame, det_by_frame, T, frames):
    """Why the misses happen: for every ground-truth group that no detection
    matched, classify the shape of the error, and separately measure whether
    recall is worse on the frames where group membership just changed."""
    classes = {"split": 0, "merged": 0, "partial": 0, "absent": 0}
    by_size = {}
    churn_rows = []
    prev_assignment = None

    for f in frames:
        g = gt_by_frame.get(f, [])
        d = det_by_frame.get(f, [])
        ftp, ffp, ffn, matches = score_frame(g, d, T)
        matched_g = {i for i, _ in matches}

        # membership churn: did anyone's groupmates change since the last frame?
        assignment = {}
        for grp in g:
            for m in grp:
                assignment[m] = frozenset(grp)
        changed = 0
        if prev_assignment is not None:
            shared = set(assignment) & set(prev_assignment)
            changed = sum(1 for m in shared if assignment[m] != prev_assignment[m])
        prev_assignment = assignment
        r = ftp / (ftp + ffn) if (ftp + ffn) else 1.0
        churn_rows.append({"frame": f, "n_changed_members": changed, "recall": r,
                           "n_gt": len(g)})

        for i, grp in enumerate(g):
            if i in matched_g:
                continue
            k = len(grp)
            by_size.setdefault(k, 0)
            by_size[k] += 1
            # which detections touch this GT group at all?
            touching = [dd for dd in d if dd & grp]
            if not touching:
                classes["absent"] += 1
            elif len(touching) >= 2 and all(len(dd & grp) < k for dd in touching):
                classes["split"] += 1
            elif any(len(dd - grp) > 0 and len(dd & grp) >= 1 for dd in touching) and \
                    any(len(dd) > k for dd in touching):
                classes["merged"] += 1
            else:
                classes["partial"] += 1

    churn = pd.DataFrame(churn_rows)
    churn_split = {}
    if len(churn):
        stable = churn[churn["n_changed_members"] == 0]
        unstable = churn[churn["n_changed_members"] > 0]
        churn_split = {
            "frames_stable": int(len(stable)),
            "frames_with_membership_change": int(len(unstable)),
            "mean_recall_stable_frames": float(stable["recall"].mean()) if len(stable) else None,
            "mean_recall_changed_frames": float(unstable["recall"].mean()) if len(unstable) else None,
        }
    return {"miss_shape": classes, "missed_gt_by_size": dict(sorted(by_size.items())),
            "churn": churn_split}


def add_facing_column(tracks, facing_mode):
    """Attach the facing channel the grouper reads, by the named source."""
    if facing_mode not in FACING_MODES:
        raise ValueError(f"facing mode must be one of {FACING_MODES}")
    if facing_mode == "none":
        return tracks, {"mode": "none", "pct_known": 0.0}
    if facing_mode == "oracle":
        out = tracks.copy()
        out["facing_rad"] = out["body_pose"]
        out["facing_conf"] = np.where(np.isfinite(out["body_pose"]), 1.0, 0.0)
        out["facing_source"] = "salsa_body_pose_annotation"
        cov = facing_mod.facing_coverage(out)
        cov["mode"] = "oracle"
        return out, cov
    out = facing_mod.add_facing(tracks)
    cov = facing_mod.facing_coverage(out)
    cov["mode"] = "motion"
    return out, cov


def run_part(salsa_root, part, eps_m=GROUPS_DEFAULT_EPS_M,
             min_samples=GROUPS_DEFAULT_MIN_SAMPLES, thresholds=(2 / 3, 1.0),
             min_cardinality=2, persistence=False,
             method=GROUPS_DEFAULT_METHOD, facing_mode="motion",
             grouper_kwargs=None):
    """Load one part, run Proxemix grouping over it, score it. Returns a dict."""
    tracks = load_salsa_tracks(salsa_root, part)
    gt = load_salsa_fformations(salsa_root, part, min_cardinality=min_cardinality)
    tracks, facing_cov = add_facing_column(tracks, facing_mode)

    clusters = detect_frame_clusters(
        tracks, eps_m=eps_m, min_samples=min_samples, x_col="x_m", y_col="y_m",
        method=method, **(grouper_kwargs or {})
    )
    if min_cardinality > 1:
        clusters = clusters[clusters["n_members"] >= min_cardinality]
    label = f"per-frame clusters ({method}, facing={facing_mode})"
    if persistence:
        gap = groups_mod.auto_gap_s(clusters)
        clusters = persistent_clusters(clusters)
        label = (f"persistence-filtered (>= {SALSA_MIN_DURATION_S:g}s, "
                 f"auto gap tolerance {gap:g}s from the data's own stride)")

    # Evaluate only on the frames SALSA actually annotated groups for — every
    # annotated timestamp, no subsampling. (All 645 / 500 of them.)
    frames = sorted(gt["frame"].unique().tolist())
    gt_by_frame = _groups_by_frame(gt)
    det_by_frame = _groups_by_frame(clusters) if len(clusters) else {}

    out = {
        "part": part, "label": PART_LABELS[part], "detector": label,
        "eps_m": eps_m, "min_samples": min_samples, "method": method,
        "grouper_kwargs": dict(grouper_kwargs or {}),
        "facing": facing_cov,
        "min_cardinality": min_cardinality,
        "n_frames_evaluated": len(frames),
        "n_gt_groups": int(len(gt)),
        "n_detected_groups": int(len(clusters)),
        "scores": {},
    }
    for T in thresholds:
        res = evaluate(gt_by_frame, det_by_frame, T, frames=frames)
        res["failure"] = failure_analysis(gt_by_frame, det_by_frame, T, frames)
        out["scores"][f"T={T:.4g}"] = res
    return out


def pool(parts_results, T_key):
    """Overall figure: pool TP/FP/FN across parts, then divide once. Never a
    joint spatial run — the two parts are not verified to share an origin."""
    tp = sum(r["scores"][T_key]["tp"] for r in parts_results)
    fp = sum(r["scores"][T_key]["fp"] for r in parts_results)
    fn = sum(r["scores"][T_key]["fn"] for r in parts_results)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc, "f1": f1}


def sweep(salsa_root, eps_values, min_samples_values=(GROUPS_DEFAULT_MIN_SAMPLES,),
          T=2 / 3, persistence=False, method=GROUPS_DEFAULT_METHOD,
          facing_mode="motion", grouper_kwargs=None):
    """Parameter exploration, reported separately from the default-parameter
    number and never substituted for it."""
    rows = []
    for ms in min_samples_values:
        for eps in eps_values:
            res = [run_part(salsa_root, p, eps_m=eps, min_samples=ms,
                            thresholds=(T,), persistence=persistence,
                            method=method, facing_mode=facing_mode,
                            grouper_kwargs=grouper_kwargs)
                   for p in SALSA_PARTS]
            overall = pool(res, f"T={T:.4g}")
            row = {"eps_m": eps, "min_samples": ms,
                   "overall_precision": overall["precision"],
                   "overall_recall": overall["recall"],
                   "overall_f1": overall["f1"]}
            for r in res:
                s = r["scores"][f"T={T:.4g}"]
                row[f"{r['part']}_f1"] = s["f1"]
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt(res):
    return (f"P={res['precision']:.3f}  R={res['recall']:.3f}  F1={res['f1']:.3f}   "
            f"(TP={res['tp']} FP={res['fp']} FN={res['fn']})")


def _pool_entries(entries):
    tp = sum(e["res"]["tp"] for e in entries)
    fp = sum(e["res"]["fp"] for e in entries)
    fn = sum(e["res"]["fn"] for e in entries)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def run_learned(args):
    """--method learned: train proxemix.learned_groups' classifier on one
    SALSA part, score on the other, both directions, for both the
    HistGradientBoostingClassifier main model and the LogisticRegression
    baseline. Never prints a number scored on the part a model trained on."""
    print(f"SALSA root: {args.salsa_root}")
    print("grouping: proxemix.learned_groups — pairwise classifier + divisive "
          "probability clustering, trained on one SALSA part and scored on "
          "the other   [NON-DEFAULT - method=learned]")
    print(f"facing source: {args.facing}"
          + ("   [UPPER BOUND - SALSA's own orientation annotation, not a "
             "Proxemix capability]" if args.facing == "oracle" else "") + "\n")

    parts_tracks, parts_gt = {}, {}
    for p in SALSA_PARTS:
        tracks = load_salsa_tracks(args.salsa_root, p)
        gt = load_salsa_fformations(args.salsa_root, p, min_cardinality=2)
        tracks, facing_cov = add_facing_column(tracks, args.facing)
        tracks = learned_mod.prepare_tracks(tracks)
        parts_tracks[p] = tracks
        parts_gt[p] = gt
        print(f"    prepared {PART_LABELS[p]} ({p}): {len(tracks)} track-rows, "
              f"{len(gt)} ground-truth group rows, "
              f"facing known for {facing_cov.get('pct_known', 0.0):.1f}% of person-frames")
    print()

    directions = [(SALSA_PARTS[0], SALSA_PARTS[1]), (SALSA_PARTS[1], SALSA_PARTS[0])]
    kinds = ("hgb", "logreg")
    by_kind = {kind: [] for kind in kinds}
    importances_by_kind = {kind: [] for kind in kinds}

    for train_p, test_p in directions:
        for kind in kinds:
            out = learned_mod.cross_part_direction(
                parts_tracks[train_p], parts_gt[train_p],
                parts_tracks[test_p], parts_gt[test_p], kind)
            gt_by_frame = _groups_by_frame(parts_gt[test_p])
            det_by_frame = _groups_by_frame(out["clusters"]) if len(out["clusters"]) else {}
            frames = sorted(parts_gt[test_p]["frame"].unique().tolist())
            res = evaluate(gt_by_frame, det_by_frame, 2 / 3, frames=frames)
            res["failure"] = failure_analysis(gt_by_frame, det_by_frame, 2 / 3, frames)
            n_det_groups = int(out["clusters"]["group_key"].nunique()) if len(out["clusters"]) else 0

            print(f"--- {kind}: train={PART_LABELS[train_p]} ({train_p}) "
                  f"-> test={PART_LABELS[test_p]} ({test_p}) ---")
            print(f"    {out['n_train_pairs']} labelled training pairs, "
                  f"tau_edge={out['tau_edge']:.2f} tau_group={out['tau_group']:.2f} "
                  f"(picked by sweeping on the TRAINING part's own ground truth)")
            print(f"    {len(frames)} annotated timestamps, {len(parts_gt[test_p])} "
                  f"ground-truth groups, {n_det_groups} detected groups "
                  f"(scored on {out['n_test_pairs']} test pairs the model never trained on)")
            print(f"    T=0.6667  {_fmt(res)}")
            print(f"              recall by true group size: " + "  ".join(
                f"k={k}:{v['recall']:.2f}(n={v['n']})"
                for k, v in res["recall_by_gt_size"].items()))
            print(f"              miss shapes: {res['failure']['miss_shape']}")
            print()

            by_kind[kind].append({"train": train_p, "test": test_p, "res": res,
                                  "tau_edge": out["tau_edge"], "tau_group": out["tau_group"],
                                  "threshold_sweep": out["threshold_sweep"],
                                  "n_train_pairs": out["n_train_pairs"],
                                  "n_test_pairs": out["n_test_pairs"]})
            imp = learned_mod.feature_importances(out["model"], kind, out["test_pairs"])
            importances_by_kind[kind].append(imp)

    for kind in kinds:
        overall = _pool_entries(by_kind[kind])
        print(f"=== OVERALL {kind} (pooled counts across both directions), "
              f"T=0.6667: {_fmt(overall)}")
    print()

    for kind in kinds:
        agg = {}
        for imp_list in importances_by_kind[kind]:
            for name, mean, _std in imp_list:
                agg.setdefault(name, []).append(mean)
        ranked = sorted(agg.items(), key=lambda kv: -np.mean(kv[1]))
        print(f"=== feature importances ({kind}, permutation on held-out test "
              f"part, ROC-AUC drop, averaged over both train/test directions) ===")
        for name, vals in ranked:
            print(f"    {name:<20} {np.mean(vals):+.4f}")
        print()

    if args.json:
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()
                        if not (isinstance(k, str) and k.startswith("_"))}
            if isinstance(o, list):
                return [clean(v) for v in o]
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        payload = {
            "salsa_root": args.salsa_root, "method": "learned",
            "facing_mode": args.facing,
            "by_kind": {
                kind: {
                    "directions": clean(by_kind[kind]),
                    "overall": clean(_pool_entries(by_kind[kind])),
                    "feature_importances_mean_roc_auc_drop": {
                        name: float(np.mean([imp[i][1] for imp in importances_by_kind[kind]
                                             for i in range(len(imp)) if imp[i][0] == name]))
                        for name in learned_mod.FEATURE_NAMES
                    },
                }
                for kind in kinds
            },
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salsa-root", default=DEFAULT_SALSA_ROOT)
    ap.add_argument("--eps-m", type=float, default=GROUPS_DEFAULT_EPS_M)
    ap.add_argument("--min-samples", type=int, default=GROUPS_DEFAULT_MIN_SAMPLES)
    ap.add_argument("--keep-singletons", action="store_true",
                    help="include cardinality-1 groups (sensitivity check)")
    ap.add_argument("--persistence", action="store_true",
                    help="score only groups surviving the >=10s episode filter")
    ap.add_argument("--method", default=GROUPS_DEFAULT_METHOD,
                    choices=("ospace", "dbscan", "learned"),
                    help="grouping algorithm; 'dbscan' is the pre-2026-10-26 "
                         "single-linkage path, kept for comparison; 'learned' "
                         "is the pairwise classifier in proxemix/learned_groups.py, "
                         "trained on one SALSA part and scored on the other")
    ap.add_argument("--build", default=GROUPS_DEFAULT_BUILD,
                    choices=("divisive", "agglomerative"),
                    help="how the o-space grouper builds candidate groups")
    ap.add_argument("--facing", default="motion", choices=FACING_MODES,
                    help="facing source: 'motion' (the shipped keyless "
                         "estimator), 'none', or 'oracle' (SALSA's own annotated "
                         "body orientation - an upper bound, not a Proxemix "
                         "capability)")
    ap.add_argument("--max-pair-m", type=float, default=None,
                    help="transitivity guard base (default from groups.py)")
    ap.add_argument("--max-pair-grow-m", type=float, default=None,
                    help="transitivity guard growth per member beyond 2")
    ap.add_argument("--circularity-min", type=float, default=None,
                    help="arrangement test threshold; 0 disables it")
    ap.add_argument("--facing-tol-deg", type=float, default=None,
                    help="o-space angular tolerance, degrees")
    ap.add_argument("--sweep", action="store_true", help="also sweep eps_m")
    ap.add_argument("--json", default=None, help="write every number to this path")
    args = ap.parse_args()

    if args.method == "learned":
        run_learned(args)
        return

    min_card = 1 if args.keep_singletons else 2
    gk = {"build": args.build}
    if args.max_pair_m is not None:
        gk["max_pair_m"] = args.max_pair_m
    if args.max_pair_grow_m is not None:
        gk["max_pair_grow_m"] = args.max_pair_grow_m
    if args.circularity_min is not None:
        gk["circularity_min"] = args.circularity_min or None
    if args.facing_tol_deg is not None:
        gk["facing_tol_rad"] = math.radians(args.facing_tol_deg)
    is_default = (args.eps_m == GROUPS_DEFAULT_EPS_M
                  and args.min_samples == GROUPS_DEFAULT_MIN_SAMPLES
                  and args.method == GROUPS_DEFAULT_METHOD
                  and args.build == GROUPS_DEFAULT_BUILD
                  and args.facing == "motion"
                  and len(gk) == 1)

    print(f"SALSA root: {args.salsa_root}")
    print(f"grouping: proxemix.groups.detect_frame_clusters("
          f"method={args.method!r}, eps_m={args.eps_m}, "
          f"min_samples={args.min_samples}"
          + "".join(f", {k}={v}" for k, v in gk.items() if k != "build") + ")"
          f"{'   [groups.py DEFAULTS]' if is_default else '   [NON-DEFAULT]'}")
    print(f"facing source: {args.facing}"
          + ("   [UPPER BOUND - SALSA's own orientation annotation, not a "
             "Proxemix capability]" if args.facing == "oracle" else ""))
    print(f"singletons: {'included' if min_card == 1 else 'excluded'}; "
          f"persistence filter: {'on' if args.persistence else 'off'}\n")

    results = [run_part(args.salsa_root, p, eps_m=args.eps_m,
                        min_samples=args.min_samples, min_cardinality=min_card,
                        persistence=args.persistence, method=args.method,
                        facing_mode=args.facing, grouper_kwargs=gk)
               for p in SALSA_PARTS]

    for r in results:
        print(f"--- {r['label']} ({r['part']}) — {r['detector']} ---")
        print(f"    {r['n_frames_evaluated']} annotated timestamps, "
              f"{r['n_gt_groups']} ground-truth groups, "
              f"{r['n_detected_groups']} detected groups")
        fc = r.get("facing") or {}
        if fc.get("mode") != "none":
            print(f"    facing: {fc.get('mode')} source, "
                  f"{fc.get('pct_known', 0.0):.1f}% of person-frames have one")
        for tk, s in r["scores"].items():
            print(f"    {tk:<9} {_fmt(s)}")
            print(f"              macro-avg over frames: "
                  f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
                  f"F1={s['macro_f1']:.3f}")
            print(f"              recall by true group size: " + "  ".join(
                f"k={k}:{v['recall']:.2f}(n={v['n']})"
                for k, v in s["recall_by_gt_size"].items()))
            print(f"              precision by detected size: " + "  ".join(
                f"k={k}:{v['precision']:.2f}(n={v['n']})"
                for k, v in s["precision_by_det_size"].items()))
            fa = s["failure"]
            print(f"              miss shapes: {fa['miss_shape']}")
            print(f"              missed true groups by size: {fa['missed_gt_by_size']}")
            c = fa["churn"]
            if c:
                print(f"              recall on stable frames "
                      f"{c['mean_recall_stable_frames']:.3f} (n={c['frames_stable']}) "
                      f"vs frames where membership just changed "
                      f"{c['mean_recall_changed_frames']:.3f} "
                      f"(n={c['frames_with_membership_change']})")
        print()

    for tk in results[0]["scores"]:
        print(f"=== OVERALL (pooled counts across both parts), {tk}: "
              f"{_fmt(pool(results, tk))}")
    print()

    sweep_df = None
    if args.sweep:
        sweep_df = sweep(args.salsa_root,
                         [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                          1.4, 1.6, 2.0, 2.5, 3.0],
                         min_samples_values=(2, 3),
                         persistence=args.persistence, method=args.method,
                         facing_mode=args.facing, grouper_kwargs=gk)
        print("=== parameter sweep (TUNED — not the out-of-the-box number), T=2/3 ===")
        print(sweep_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        best = sweep_df.loc[sweep_df["overall_f1"].idxmax()]
        print(f"\n    best overall F1 {best['overall_f1']:.3f} at "
              f"eps_m={best['eps_m']}, min_samples={int(best['min_samples'])}")

    if args.json:
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()
                        if not (isinstance(k, str) and k.startswith("_"))}
            if isinstance(o, list):
                return [clean(v) for v in o]
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        payload = {
            "salsa_root": args.salsa_root,
            "eps_m": args.eps_m, "min_samples": args.min_samples,
            "method": args.method, "facing_mode": args.facing,
            "grouper_kwargs": {k: v for k, v in gk.items()},
            "groups_py_defaults_used": is_default,
            "singletons_included": min_card == 1,
            "persistence_filter": args.persistence,
            "salsa_fps": SALSA_FPS,
            "parts": clean(results),
            "overall": {tk: pool(results, tk) for tk in results[0]["scores"]},
        }
        if sweep_df is not None:
            payload["eps_sweep"] = sweep_df.to_dict(orient="records")
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
