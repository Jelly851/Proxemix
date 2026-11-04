"""Can tiling's recall be kept without its precision collapse? Swept offline.

The measured problem, SALSA cam1 cocktail party, 16 frames, 2026-11-04:

    full frame   recall 64.6%   precision 65.3%   F1 0.649
    16 tiles     recall 74.5%   precision 29.8%   F1 0.426

Tiling buys 9.9 points of recall and loses 35 points of precision, so the
combined score is **worse** than not tiling at all. Raw detection count said
the splits were paying; measured against annotations most of the extra boxes
are not people. That is the whole reason this sweep exists.

Every box from that run is on disk, so merge threshold, confidence floor and
shape sanity can be swept against the ground truth for free. No API calls.

    py -m proxemix.tiling_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import sys

import pandas as pd

from proxemix import sam_video
from proxemix.salsa_adapter import load_salsa_tracks
from proxemix.salsa_camera import SalsaCamera
from proxemix.salsa_sam_facing import (
    DEFAULT_MAX_MATCH_M,
    DEFAULT_PART,
    DEFAULT_SALSA_ROOT,
    match_to_participants,
)
from proxemix.tiling import merge_tile_detections

RAW_DEFAULT = "outputs/tiling_calibration/salsa_raw_boxes_d2_16f.json"
OUT_PATH = "outputs/tiling_calibration/salsa_threshold_sweep.json"


def evaluate(rows, gt, cam, max_match_m=DEFAULT_MAX_MATCH_M):
    """recall, precision and F1 over annotated person-frames."""
    if not rows:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "boxes": 0}
    table = pd.DataFrame([
        {"concept": sam_video.DEFAULT_PERSON_CONCEPT, "frame": int(f),
         "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
         "conf": b[4] if len(b) > 4 else 0.0} for f, b in rows])
    matched, _ = match_to_participants(table, gt, cam, max_match_m)
    covered = set(zip(matched.loc[matched.track_id.notna(), "track_id"],
                      matched.loc[matched.track_id.notna(), "frame"]))
    gt_frames = gt[gt.frame.isin(set(table["frame"]))]
    recall = len(covered) / max(1, len(gt_frames))
    precision = len(covered) / max(1, len(table))
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return {"recall": round(recall, 4), "precision": round(precision, 4),
            "f1": round(f1, 4), "boxes": int(len(table)),
            "matched": len(covered), "gt": int(len(gt_frames))}


def touches_tile_edge(box, tile, frame_w, frame_h, margin=3):
    """True when the box runs into a cut line rather than ending at the person.

    This is the filter the precision collapse asked for. A person standing
    across a tile boundary is detected in that tile as the visible half, so
    the box ends at the cut. Everything downstream then reads the bottom of
    that box as their feet, the homography puts their feet metres from where
    they are, and a perfectly real detection becomes an unmatchable one.

    A tile side that lies on the frame border is **not** a cut — the person
    really is cut off there in the source footage, and the same truncation is
    present in the untiled baseline, so excluding it would change what is
    being compared.

    Overlap is what makes dropping these safe: a person cut in one tile sits
    whole inside the neighbouring tile, so the detection is not lost, only the
    broken copy of it.
    """
    x1, y1, x2, y2 = box[:4]
    tx1, ty1, tx2, ty2 = tile
    if tx1 > 0 and x1 <= tx1 + margin:
        return True
    if ty1 > 0 and y1 <= ty1 + margin:
        return True
    if tx2 < frame_w and x2 >= tx2 - margin:
        return True
    if ty2 < frame_h and y2 >= ty2 - margin:
        return True
    return False


def filtered_rows(raw, conf_min, merge_iou, drop_wide, min_h,
                  drop_edge=False, frame=(1024, 768), containment=None):
    """Apply one setting and return [(frame, box), ...] after merging.

    Merging happens per frame, because two boxes on the same person only ever
    occur within one frame — merging across frames would delete real people
    who stood still.
    """
    fw, fh = frame
    by_frame = {}
    for r in raw:
        h = r["y2"] - r["y1"]
        w = r["x2"] - r["x1"]
        if r["conf"] < conf_min:
            continue
        if h < min_h:
            continue
        if drop_wide and w > h:
            continue
        if drop_edge and r.get("tile") and touches_tile_edge(
                (r["x1"], r["y1"], r["x2"], r["y2"]), r["tile"], fw, fh):
            continue
        by_frame.setdefault(r["frame"], []).append(
            [r["x1"], r["y1"], r["x2"], r["y2"], r["conf"]])

    out = []
    for frame, boxes in by_frame.items():
        if containment is None:
            merged, _ = merge_tile_detections([(None, boxes)],
                                              iou_threshold=merge_iou)
        else:
            merged = merge_by_containment(boxes, iou_threshold=merge_iou,
                                          containment=containment)
        for b in merged:
            out.append((frame, b))
    return out


def merge_by_containment(boxes, iou_threshold=0.55, containment=0.70):
    """One frame's boxes, duplicates dropped on overlap or containment.

    The rule now lives in `tiling.merge_tile_detections`, which is where the
    pipeline reads it from — this stays as the name the sweep and the scale
    evaluation already call, so one behaviour has one home.
    """
    kept, _ = merge_tile_detections([(None, boxes)],
                                    iou_threshold=iou_threshold,
                                    containment_threshold=containment)
    return kept


def run(raw_path=RAW_DEFAULT, part=DEFAULT_PART):
    if not os.path.exists(raw_path):
        print(f"no raw boxes at {raw_path} — run tiling_salsa_eval first",
              file=sys.stderr)
        return 1
    blob = json.loads(pathlib.Path(raw_path).read_text(encoding="utf-8"))
    raw = blob["boxes"]
    cam = SalsaCamera.cam(1, os.path.join(DEFAULT_SALSA_ROOT, "calibration"))
    gt = load_salsa_tracks(DEFAULT_SALSA_ROOT, part)
    print(f"{len(raw)} raw boxes over {blob['n_frames']} frames, "
          f"{blob['n_tiles'] if 'n_tiles' in blob else len(blob['tiles'])} tiles")

    frame = tuple(blob.get("frame") or (1024, 768))
    confs = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]
    ious = [0.15, 0.25, 0.35, 0.45, 0.55]
    wides = [False, True]
    min_hs = [0, 30]
    edges = [False, True]
    contains = [None, 0.85, 0.70, 0.55, 0.40]

    rows = []
    best = None
    for conf, miou, wide, mh, edge, cont in itertools.product(
            confs, ious, wides, min_hs, edges, contains):
        kept = filtered_rows(raw, conf, miou, wide, mh, drop_edge=edge,
                             frame=frame, containment=cont)
        m = evaluate(kept, gt, cam)
        m.update({"conf_min": conf, "merge_iou": miou,
                  "drop_wider_than_tall": wide, "min_height_px": mh,
                  "drop_tile_edge_boxes": edge, "containment": cont})
        rows.append(m)
        if best is None or m["f1"] > best["f1"]:
            best = m

    rows.sort(key=lambda r: -r["f1"])
    print("\ntop 10 by F1:")
    print(f"  {'conf':>5} {'iou':>5} {'wide':>5} {'minh':>5} {'edge':>5} "
          f"{'cont':>5} {'recall':>7} {'prec':>6} {'F1':>6} {'boxes':>6}")
    for r in rows[:12]:
        print(f"  {r['conf_min']:>5} {r['merge_iou']:>5} "
              f"{str(r['drop_wider_than_tall']):>5} {r['min_height_px']:>5} "
              f"{str(r['drop_tile_edge_boxes']):>5} {str(r['containment']):>5} "
              f"{r['recall']:>7} {r['precision']:>6} {r['f1']:>6} {r['boxes']:>6}")

    # The comparison that decides whether tiling ships at all.
    baseline = {"recall": 0.646, "precision": 0.653, "f1": 0.649,
                "source": "full-frame arm of salsa_tiling_eval_d2_16f.json"}
    print(f"\nfull-frame baseline: recall {baseline['recall']} "
          f"precision {baseline['precision']} F1 {baseline['f1']}")
    print(f"best tiled setting:  recall {best['recall']} "
          f"precision {best['precision']} F1 {best['f1']}")
    verdict = ("tiling wins on F1" if best["f1"] > baseline["f1"]
               else "tiling still loses on F1 — recall is bought with too much "
                    "precision")
    print(f"VERDICT: {verdict}")

    # Also report the best setting at each recall floor, because recall is what
    # the product is short of — a setting that beats the baseline's F1 while
    # also beating its recall is the only one worth shipping.
    print("\nbest F1 at each recall floor:")
    for floor in (0.65, 0.68, 0.70, 0.72, 0.74):
        cand = [r for r in rows if r["recall"] >= floor]
        if cand:
            c = max(cand, key=lambda r: r["f1"])
            print(f"  recall >= {floor}: F1 {c['f1']} (recall {c['recall']}, "
                  f"prec {c['precision']}, conf {c['conf_min']}, "
                  f"iou {c['merge_iou']}, wide {c['drop_wider_than_tall']}, "
                  f"minh {c['min_height_px']}, edge {c['drop_tile_edge_boxes']}, "
                  f"cont {c['containment']})")
        else:
            print(f"  recall >= {floor}: unreachable with any setting swept")

    out = pathlib.Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"raw_path": raw_path, "n_raw_boxes": len(raw),
         "n_frames": blob["n_frames"], "baseline_full_frame": baseline,
         "best_by_f1": best, "verdict": verdict, "all_settings": rows},
        indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="offline threshold sweep")
    ap.add_argument("--raw", default=RAW_DEFAULT)
    args = ap.parse_args()
    return run(raw_path=args.raw)


if __name__ == "__main__":
    sys.exit(main())
