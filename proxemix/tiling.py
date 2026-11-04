"""Adaptive tiling — the frame is cut where people are small, not on a grid.

Why this module exists
----------------------
SAM resizes whatever it is given to a fixed internal resolution, so a person
who occupies 2% of a 1440-tall frame arrives at the detector as a handful of
pixels and is missed. Cutting the frame into pieces and detecting in each one
makes that same person occupy a far larger share of what the detector sees.

Measured on our own footage before any of this was written: plaza people run
a median of 112 pixels tall with the smallest 5% at 28, and mall people 185
with the smallest at 55. Those smallest ones are the misses.

The three design decisions, and the evidence for each
-----------------------------------------------------
1. **Subdivide recursively, never on a fixed grid.** Density across our own
   frames runs 3.4x uneven (plaza quadrants: 822/620/2092/1539) and 13x
   uneven (mall: 275/581/2608/3653). One split count for the whole frame is
   either too coarse where the crowd is or wasted where it is not.

2. **Never skip a region, however empty it looks.** ROI-gated slicing —
   running a cheap proposer and skipping tiles it calls background — is
   published (arXiv 2608.23923) and it *lost* accuracy on a real dataset:
   mAP@0.5 of 0.660 against 0.757 for full slicing, while also running
   slower on average. It wins only on genuinely sparse scenes, and none of
   our frames have an empty quadrant. So a sparse region simply stops
   subdividing; it is never dropped.

3. **Overlap tiles by 20%.** Slicing-aided inference measures 25% overlap as
   worth up to +1.7 AP (arXiv 2202.06934), and the diminishing-returns knee
   sits around 20% — past it, 40% overlap buys about 2.5 points of recall for
   a large compute increase. 20% is the knee rather than the maximum.

The cost argument, which is the whole reason this is adaptive
-------------------------------------------------------------
A plan is computed **once per clip** from the headcount call the pipeline
already pays for, then reused for every frame. So adapting costs nothing per
frame: the planning information is a by-product of a call already made.

Run it
------
    py -m proxemix.tiling --selftest
    py -m proxemix.tiling --calibrate data/footage/frames_plaza/frame_00045.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

DEFAULT_OVERLAP = 0.20      # the measured knee, not the maximum
DEFAULT_MAX_DEPTH = 2       # depth 2 = up to 16 leaf regions
DEFAULT_MIN_TILE_PX = 320   # below this a tile is mostly context, not subject
MERGE_IOU = 0.55            # two boxes this similar across a seam are one person

# THE RULE IS A CEILING ON HOW MUCH OF A TILE ONE PERSON MAY FILL, and every
# part of it was measured on 2026-11-04 rather than chosen.
#
# What decides detectability is the share of the tile a person occupies, not
# their pixel height — SAM resizes whatever it is sent to a fixed internal
# resolution. Plaza frame_00090 shows the share is the invariant: the pixel
# floor of what gets found moves with the tile (61 px at full frame, 31 px at
# 2x2) while the share stays near 4-5%.
#
# But the first version of this rule only had a floor, and a floor alone is
# wrong in the expensive direction. Scored against SALSA's annotations, the
# same 12 frames, 2x2 against one full-frame call:
#
#   people as sent   grid   recall          precision       F1
#   63 px            1x1    0.623           0.683           0.652
#   63 px            2x2    0.676 (+5.3pp)  0.667           0.672
#   190 px           1x1    0.647           0.673           0.660
#   190 px           2x2    0.691 (+4.3pp)  0.691           0.691
#   190 px           4x4    0.657 (+1.0pp)  0.383           0.484
#
# 4x4 is the finding that shaped the rule. Tiles there are 256x192 while the
# people are 190 px tall, so **a person is as tall as the tile** and is cut in
# every tile they appear in. Recall barely moves and precision halves.
#
# Writing the ratio of person height to tile height for each row: full frame
# 0.25, 2x2 0.49, 4x4 0.99. The win sits at roughly a half and the collapse at
# roughly one. So the rule is a ceiling — **split only while the people would
# still be at most half the tile afterwards** — and since a split halves the
# tile, that means splitting while they are at most a quarter of it now.
#
# It reproduces all three measured cases and the plaza calibration: plaza at
# 0.078 splits twice to 0.31 (and plaza measured 3x3 better than 2x2), SALSA at
# 0.25 splits exactly once to 0.49, and nothing splits past that.
MAX_PERSON_SHARE_AFTER_SPLIT = 0.5

# Duplicate removal has to use containment, not overlap. A person cut at a
# tile edge shares only part of their area with the whole-body box in the
# neighbouring tile, so their overlap score is small while the cut piece sits
# almost entirely *inside* the whole box. Measured on the 16-tile SALSA run:
# overlap-only merging left precision at 0.298, and adding containment lifted
# it to 0.413 on the same boxes. `tiling_sweep.merge_by_containment` holds it.
DEFAULT_CONTAINMENT = 0.70


# --- geometry ---------------------------------------------------------------

class Region:
    """A rectangle in full-frame pixel coordinates, and its place in the tree."""

    __slots__ = ("x1", "y1", "x2", "y2", "depth")

    def __init__(self, x1, y1, x2, y2, depth=0):
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
        self.depth = int(depth)

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    def contains_center(self, box):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        return self.x1 <= cx < self.x2 and self.y1 <= cy < self.y2

    def quarters(self):
        """Four children, split at the midpoint of each axis."""
        mx = self.x1 + self.w // 2
        my = self.y1 + self.h // 2
        d = self.depth + 1
        return [Region(self.x1, self.y1, mx, my, d),
                Region(mx, self.y1, self.x2, my, d),
                Region(self.x1, my, mx, self.y2, d),
                Region(mx, my, self.x2, self.y2, d)]

    def padded(self, overlap, frame_w, frame_h):
        """The region grown by `overlap` on every side, clipped to the frame.

        Overlap exists so a person standing on a cut line is whole in at least
        one tile. Growing rather than shrinking is deliberate: the union of
        padded tiles still covers the frame exactly once at minimum.
        """
        px = int(round(self.w * overlap / 2.0))
        py = int(round(self.h * overlap / 2.0))
        return (max(0, self.x1 - px), max(0, self.y1 - py),
                min(frame_w, self.x2 + px), min(frame_h, self.y2 + py))

    def as_dict(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "depth": self.depth}

    def __repr__(self):
        return f"Region({self.x1},{self.y1},{self.x2},{self.y2},d{self.depth})"


def percentile(values, q):
    """The q-th percentile by linear interpolation, with no numpy import.

    Written out rather than imported because this module is the one the
    server may call per request, and numpy is already the heaviest thing in
    the package's import graph.
    """
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def plan_tiles(boxes, frame_w, frame_h,
               max_share_after_split=MAX_PERSON_SHARE_AFTER_SPLIT,
               overlap=DEFAULT_OVERLAP, max_depth=DEFAULT_MAX_DEPTH,
               min_tile_px=DEFAULT_MIN_TILE_PX, size_quantile=0.50,
               min_boxes_to_split=2):
    """Boxes from one frame -> the tile plan for the whole clip.

    `boxes` are [x1, y1, x2, y2] in full-frame pixels, from the headcount call
    the pipeline already makes. The rule and its evidence live in
    `MAX_PERSON_SHARE_AFTER_SPLIT` above.

    The rule, stated once: **split while the people would still be at most
    half the tile afterwards.** A split halves the tile, so their share
    doubles; refusing to cross the ceiling is what stops people being cut into
    fragments in every tile they appear in.

    The median rather than a low quantile, deliberately: the ceiling is about
    the people most at risk of being truncated, which are the larger ones, so
    the typical person is the right thing to protect. The first version of this
    function used the 25th percentile, which optimises for the small tail and
    is how 4x4 looked like a good idea.

    Returns `{"regions": [...], "tiles": [...], "stats": {...}}` where every
    region is a leaf and the leaves tile the frame exactly.
    """
    frame_w, frame_h = int(frame_w), int(frame_h)
    root = Region(0, 0, frame_w, frame_h, 0)
    leaves = []
    reasons = {"would_fill_tile": 0, "max_depth": 0, "too_small": 0,
               "too_few": 0}

    def recurse(region):
        inside = [b for b in boxes if region.contains_center(b)]
        h_q = percentile([b[3] - b[1] for b in inside], size_quantile) if inside else 0.0
        share_now = (h_q / region.h) if region.h else 0.0

        if inside and share_now * 2.0 > max_share_after_split:
            reasons["would_fill_tile"] += 1
            leaves.append(region)
            return
        if region.depth >= max_depth:
            reasons["max_depth"] += 1
            leaves.append(region)
            return
        if min(region.w, region.h) // 2 < min_tile_px:
            reasons["too_small"] += 1
            leaves.append(region)
            return
        if len(inside) < min_boxes_to_split:
            # A region with nobody in it is still detected, never skipped —
            # it just has no evidence that splitting it would help.
            reasons["too_few"] += 1
            leaves.append(region)
            return
        for child in region.quarters():
            recurse(child)

    recurse(root)
    heights = [b[3] - b[1] for b in boxes]
    return {
        "regions": [r.as_dict() for r in leaves],
        "tiles": [r.padded(overlap, frame_w, frame_h) for r in leaves],
        "stats": {
            "n_tiles": len(leaves),
            "max_depth_used": max((r.depth for r in leaves), default=0),
            "overlap": overlap,
            "max_share_after_split": max_share_after_split,
            "stop_reasons": reasons,
            "n_boxes_planned_from": len(boxes),
            "person_height_median": round(percentile(heights, 0.50), 1),
            "person_share_of_frame": (round(percentile(heights, 0.50) / frame_h, 4)
                                      if frame_h else None),
            "frame": [frame_w, frame_h],
        },
    }


# --- merging detections back together ---------------------------------------

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(a, b):
    """Intersection over the area of the *smaller* box.

    The signal overlap misses. A person cut at a tile edge shares only part of
    their area with the whole-body box from the neighbouring tile, so overlap
    reads them as different people while the cut piece sits almost entirely
    inside the whole one. This reads that case at near 1.0 however different
    the two sizes are.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def merge_tile_detections(per_tile, iou_threshold=MERGE_IOU,
                          containment_threshold=DEFAULT_CONTAINMENT):
    """Detections from overlapping tiles -> one list, duplicates dropped.

    `per_tile` is a list of `(tile_rect, [box, ...])` with every box already
    in full-frame coordinates. Highest confidence wins a duplicate, which
    matters because the copy from the tile where the person sat near the edge
    is usually the worse one.

    **Two signals, either of which makes a duplicate**, and the second one is
    load-bearing rather than belt-and-braces: on the 16-tile SALSA run,
    overlap alone left precision at 0.298 and adding containment lifted it to
    0.413 on the identical boxes. Pass `containment_threshold=None` to get the
    old overlap-only behaviour.

    Boxes carry an optional 5th element, confidence.
    """
    flat = []
    for _, boxes in per_tile:
        flat.extend(boxes)
    flat.sort(key=lambda b: -(b[4] if len(b) > 4 else 0.0))

    kept, dropped = [], 0
    for box in flat:
        duplicate = False
        for k in kept:
            if iou(box, k) >= iou_threshold:
                duplicate = True
                break
            if (containment_threshold is not None
                    and containment(box, k) >= containment_threshold):
                duplicate = True
                break
        if duplicate:
            dropped += 1
            continue
        kept.append(box)
    return kept, dropped


# --- progressive refinement, the rule that actually works -------------------
# Predicting the split from person size failed for a reason worth keeping:
# **the planner can only see the people the detector already found, and the
# people who matter are the ones it missed.** Measured 2026-11-04 on
# frames_plaza/frame_00090 — planning from the full-frame call's 65 boxes
# stopped at 4 tiles and reached 103 people, while a blind uniform 3x3 reached
# 118. The prediction was confidently wrong in the one direction that costs
# recall.
#
# So the rule is empirical instead: **split a region again only when the last
# split of that region actually found more people.** Yield is observed rather
# than forecast, every call is spent where the previous call proved there was
# something to find, and a region that stops paying stops being split. This is
# the cost-effectiveness Jayden asked for, expressed as a stopping rule.

DEFAULT_YIELD_GAIN = 0.15   # a split must find 15% more people to earn another


def refine_by_yield(image_path, client, concept="person",
                    max_depth=3, yield_gain=DEFAULT_YIELD_GAIN,
                    overlap=DEFAULT_OVERLAP, min_tile_px=DEFAULT_MIN_TILE_PX,
                    min_people_to_split=3, frame_size=None, verbose=True):
    """Detect on one frame, splitting only where splitting keeps paying.

    Returns `{"boxes": [...], "n_calls": int, "rounds": [...], "tiles": [...]}`
    where `tiles` is the plan that earned its keep — reusable for every other
    frame in the clip, which is where the saving comes from.

    A region's yield is counted over **its own area only**, so a busy corner
    is judged on what it contains rather than on the frame's total.
    """
    from PIL import Image

    if frame_size is None:
        frame_size = Image.open(image_path).size
    fw, fh = frame_size
    tmp_dir = tempfile.mkdtemp(prefix="proxemix_refine_")

    def detect(region):
        rect = region.padded(overlap, fw, fh)
        per_tile, fails = detect_in_tiles(image_path, [rect], client,
                                          concept=concept, tmp_dir=tmp_dir)
        boxes = per_tile[0][1]
        # Only boxes whose centre lands inside the unpadded region count for
        # this region, otherwise the overlap would be counted twice and every
        # split would look like it paid off.
        owned = [b for b in boxes if region.contains_center(b)]
        return owned, len(fails)

    root = Region(0, 0, fw, fh, 0)
    root_boxes, fails = detect(root)
    n_calls = 1
    rounds = [{"depth": 0, "n_regions": 1, "n_people": len(root_boxes),
               "calls": 1, "failures": fails}]
    if verbose:
        print(f"  depth 0: {len(root_boxes)} people, 1 call", flush=True)

    # Each entry is a region plus what it yielded at the current depth.
    frontier = [(root, root_boxes)]
    final = []

    for depth in range(1, max_depth + 1):
        next_frontier, round_calls, round_fails, round_people = [], 0, 0, 0
        for region, before in frontier:
            splittable = (min(region.w, region.h) // 2 >= min_tile_px
                          and len(before) >= min_people_to_split)
            if not splittable:
                final.append((region, before))
                continue

            children, child_boxes, c_calls, c_fails = [], [], 0, 0
            for child in region.quarters():
                boxes, f = detect(child)
                c_calls += 1
                c_fails += f
                children.append((child, boxes))
                child_boxes.extend(boxes)

            round_calls += c_calls
            round_fails += c_fails
            merged, _ = merge_tile_detections([(None, child_boxes)])
            gain = (len(merged) - len(before)) / max(1, len(before))

            if gain >= yield_gain:
                # It paid. Keep the children and consider splitting them again.
                next_frontier.extend(children)
                round_people += len(merged)
            else:
                # It did not pay. Keep whichever result is larger — the calls
                # are already spent, so throwing the better answer away to
                # honour the stopping rule would be spending for nothing.
                if len(merged) > len(before):
                    final.extend(children)
                    round_people += len(merged)
                else:
                    final.append((region, before))
                    round_people += len(before)

        rounds.append({"depth": depth, "n_regions_split": len(frontier),
                       "n_regions_kept": len(next_frontier),
                       "calls": round_calls, "failures": round_fails,
                       "n_people_this_depth": round_people})
        if verbose:
            print(f"  depth {depth}: {round_calls} calls, "
                  f"{len(next_frontier)} regions earned another split, "
                  f"{round_people} people found at this depth", flush=True)
        n_calls += round_calls
        frontier = next_frontier
        if not frontier:
            break

    final.extend(frontier)
    all_boxes = []
    for _, boxes in final:
        all_boxes.extend(boxes)
    merged, dropped = merge_tile_detections([(None, all_boxes)])

    return {
        "boxes": merged,
        "n_people": len(merged),
        "n_calls": n_calls,
        "duplicates_dropped": dropped,
        "rounds": rounds,
        "tiles": [r.padded(overlap, fw, fh) for r, _ in final],
        "regions": [r.as_dict() for r, _ in final],
        "frame": [fw, fh],
        "yield_gain": yield_gain,
        "spent_usd": round(client.spent_usd, 4),
    }


# --- the calibration experiment ---------------------------------------------
# What is being measured: how many more people the detector finds as the frame
# is cut finer, and how small the newly-found people are. That curve is what
# sets `target_height` — everything above is geometry, and this is the number.

def crop_to_temp(image_path, rect, out_dir):
    from PIL import Image
    img = Image.open(image_path)
    crop = img.crop((int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
    path = os.path.join(
        out_dir, f"crop_{int(rect[0])}_{int(rect[1])}_{int(rect[2])}_{int(rect[3])}.jpg")
    crop.save(path, quality=95)
    return path, crop.size


def uniform_grid(frame_w, frame_h, n, overlap=DEFAULT_OVERLAP):
    """An n x n grid of padded tiles — the fixed strategy, for comparison."""
    out = []
    for row in range(n):
        for col in range(n):
            r = Region(frame_w * col // n, frame_h * row // n,
                       frame_w * (col + 1) // n, frame_h * (row + 1) // n, 1)
            out.append(r.padded(overlap, frame_w, frame_h))
    return out


def detect_in_tiles(image_path, tiles, client, concept="person", tmp_dir=None):
    """Run one image call per tile, mapping every box back to full-frame pixels.

    A failed tile is recorded and survived rather than raised: losing one tile
    of nine should not lose the measurement, and the caller is told how many
    failed so a thin result reads as thin.
    """
    from proxemix.sam_video import parse_segmentation

    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="proxemix_tiles_")
    os.makedirs(tmp_dir, exist_ok=True)
    per_tile, failures = [], []

    for rect in tiles:
        path, (cw, ch) = crop_to_temp(image_path, rect, tmp_dir)
        try:
            text = client.segment_image(path, concept=concept)
            parsed = parse_segmentation(text)
        except Exception as e:
            failures.append({"tile": list(rect), "error": f"{type(e).__name__}: {e}"[:200]})
            per_tile.append((rect, []))
            continue

        boxes = []
        for dets in parsed.values():
            for det in dets:
                bx = det.get("bbox")
                if not bx:
                    continue
                # SAM reports in the coordinate space of what it was sent, so
                # scale by the crop's real size over the reported size before
                # offsetting. Assuming they are equal is how a box lands in
                # the wrong place when the API normalises or pads.
                rw = float(det.get("frame_width") or cw) or cw
                rh = float(det.get("frame_height") or ch) or ch
                sx, sy = cw / rw, ch / rh
                boxes.append([bx[0] * sx + rect[0], bx[1] * sy + rect[1],
                              bx[2] * sx + rect[0], bx[3] * sy + rect[1],
                              float(det.get("confidence") or 0.0)])
        per_tile.append((rect, boxes))

    return per_tile, failures


def calibrate(image_path, client, levels=(1, 2, 3), concept="person",
              out_dir="outputs/tiling_calibration"):
    """Detect on one frame at several uniform grid levels and compare.

    Uniform grids rather than adaptive ones, deliberately: this run exists to
    find the person height at which detection starts working, and a uniform
    grid gives a clean size-versus-level curve with no planning policy mixed
    into it. The adaptive planner is then fed the number this produces.
    """
    from PIL import Image

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fw, fh = Image.open(image_path).size
    tmp_dir = tempfile.mkdtemp(prefix="proxemix_calib_")

    results = {"image": str(image_path), "frame": [fw, fh], "concept": concept,
               "levels": []}
    for n in levels:
        tiles = [(0, 0, fw, fh)] if n == 1 else uniform_grid(fw, fh, n)
        per_tile, failures = detect_in_tiles(image_path, tiles, client,
                                             concept=concept, tmp_dir=tmp_dir)
        merged, dropped = merge_tile_detections(per_tile)
        heights = [b[3] - b[1] for b in merged]
        results["levels"].append({
            "grid": f"{n}x{n}",
            "n_tiles": len(tiles),
            "n_calls": len(tiles),
            "n_detections": len(merged),
            "n_duplicates_dropped": dropped,
            "n_tile_failures": len(failures),
            "failures": failures[:4],
            "height_p05": round(percentile(heights, 0.05), 1),
            "height_p25": round(percentile(heights, 0.25), 1),
            "height_median": round(percentile(heights, 0.50), 1),
            "spent_usd": round(client.spent_usd, 4),
        })
        print(f"  {n}x{n}: {len(merged)} people "
              f"({len(tiles)} calls, {dropped} dupes dropped, "
              f"{len(failures)} tile failures), "
              f"heights p05 {results['levels'][-1]['height_p05']} "
              f"median {results['levels'][-1]['height_median']}, "
              f"${client.spent_usd:.4f} so far", flush=True)

    base = results["levels"][0]["n_detections"] if results["levels"] else 0
    for lvl in results["levels"]:
        lvl["vs_full_frame"] = (round(lvl["n_detections"] / base, 3)
                                if base else None)
    path = out / (pathlib.Path(image_path).stem + "_calibration.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  written: {path}")
    return results


# --- selftest ---------------------------------------------------------------

def selftest():
    # Geometry: leaves tile the frame exactly, with no gaps and no overlaps.
    boxes = [[100, 100, 130, 160]] * 8                       # tiny, bottom-left
    plan = plan_tiles(boxes, 1000, 1000, min_tile_px=50)
    leaves = [Region(**r) for r in plan["regions"]]
    area = sum(r.w * r.h for r in leaves)
    assert area == 1000 * 1000, (area, plan["stats"])

    # A frame of big people needs no split at all — they already fill too much
    # of it, and splitting would cut them.
    big = [[10, 10, 300, 610], [400, 10, 700, 620]]
    flat = plan_tiles(big, 1000, 1000)
    assert flat["stats"]["n_tiles"] == 1, flat["stats"]

    # --- the three measured cases, as regression tests ----------------------
    # SALSA: 190 px people in a 768-tall frame, where 2x2 measured better than
    # both 1x1 and 4x4. The rule must choose exactly 4 tiles.
    salsa = [[100 + i * 30, 300, 160 + i * 30, 490] for i in range(17)]
    p_salsa = plan_tiles(salsa, 1024, 768, max_depth=3, min_tile_px=100)
    assert p_salsa["stats"]["n_tiles"] == 4, p_salsa["stats"]

    # Plaza: 112 px people in a 1440-tall frame, where 3x3 measured better than
    # 2x2. The rule must go deeper here than on SALSA.
    plaza = [[100 + i * 40, 600, 140 + i * 40, 712] for i in range(30)]
    p_plaza = plan_tiles(plaza, 2560, 1440, max_depth=3, min_tile_px=100)
    assert p_plaza["stats"]["n_tiles"] > p_salsa["stats"]["n_tiles"], \
        (p_salsa["stats"]["n_tiles"], p_plaza["stats"]["n_tiles"])

    # And nothing ever splits to where a person fills the tile, which is the
    # 4x4 failure the rule exists to prevent.
    for plan_x, boxes_x in ((p_plaza, plaza), (p_salsa, salsa)):
        for r in plan_x["regions"]:
            region = Region(**r)
            inside = [b for b in boxes_x if region.contains_center(b)]
            if inside:
                med = percentile([b[3] - b[1] for b in inside], 0.5)
                assert med / region.h <= MAX_PERSON_SHARE_AFTER_SPLIT + 1e-9, \
                    (med, region.h, med / region.h)

    # Scale invariance: the same people-to-frame ratio gives the same depth at
    # any resolution. An absolute pixel target would fail this.
    ratio_a = plan_tiles([[i * 10, 0, i * 10 + 20, 60] for i in range(30)],
                         1000, 1000, max_depth=4, min_tile_px=10)
    ratio_b = plan_tiles([[i * 20, 0, i * 20 + 40, 120] for i in range(30)],
                         2000, 2000, max_depth=4, min_tile_px=10)
    assert (ratio_a["stats"]["max_depth_used"]
            == ratio_b["stats"]["max_depth_used"]), (ratio_a["stats"],
                                                     ratio_b["stats"])

    # Uneven density: small people in one corner only, big elsewhere. The
    # corner must end up with more tiles than the rest — the whole point.
    mixed = [[50 + i, 50, 58 + i, 70] for i in range(20)] \
        + [[1200, 1200, 1500, 1800], [1600, 1200, 1900, 1800]]
    plan_m = plan_tiles(mixed, 2000, 2000, max_depth=2, min_tile_px=100)
    small_corner = [Region(**r) for r in plan_m["regions"]
                    if r["x2"] <= 1000 and r["y2"] <= 1000]
    big_corner = [Region(**r) for r in plan_m["regions"]
                  if r["x1"] >= 1000 and r["y1"] >= 1000]
    assert len(small_corner) > len(big_corner), (len(small_corner), len(big_corner))

    # No region is ever dropped, however empty: an empty frame is one tile
    # covering everything, not zero tiles.
    empty = plan_tiles([], 800, 600, min_tile_px=50)
    assert empty["stats"]["n_tiles"] == 1 and empty["regions"][0]["x2"] == 800

    # Overlap grows the tile and clips at the frame edge.
    pad = Region(0, 0, 500, 500).padded(0.20, 1000, 1000)
    assert pad == (0, 0, 550, 550), pad

    # Merging: the same person seen in two tiles survives once, at the higher
    # confidence, and two genuinely different people both survive.
    dup = [((0, 0, 100, 100), [[10, 10, 50, 90, 0.6]]),
           ((50, 0, 150, 100), [[11, 11, 51, 91, 0.9], [120, 10, 140, 90, 0.8]])]
    kept, dropped = merge_tile_detections(dup)
    assert len(kept) == 2 and dropped == 1, (kept, dropped)
    assert kept[0][4] == 0.9, kept[0]

    # Percentile matches the obvious hand-computed answers.
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.9) == 5.0
    assert percentile([], 0.5) == 0.0

    print("tiling self-test PASSED — leaves tile the frame exactly, big people "
          "are left unsplit, SALSA's 190px people get exactly the 4 tiles that "
          "measured best, plaza's 112px people go deeper, no region ever splits "
          "to where a person fills the tile, the same ratio gives the same "
          "depth at any resolution, the dense corner gets more tiles than the "
          "sparse one, empty regions are kept rather than dropped, overlap "
          "clips at the edge, and a person seen twice across a seam survives "
          "once at the higher confidence.")
    return 0


def compare_adaptive(image_path, client, uniform_n=3, concept="person",
                     max_share_after_split=MAX_PERSON_SHARE_AFTER_SPLIT,
                     max_depth=DEFAULT_MAX_DEPTH,
                     out_dir="outputs/tiling_calibration"):
    """The decisive test: adaptive against a uniform grid, counted per call.

    Adaptive tiling is only worth its complexity if it finds at least as many
    people as a uniform grid for fewer calls, so that is what this measures —
    people per call, not people. The planner is fed the boxes from one
    full-frame call, which is exactly what the pipeline will have.
    """
    from PIL import Image

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fw, fh = Image.open(image_path).size
    tmp_dir = tempfile.mkdtemp(prefix="proxemix_cmp_")

    # Step 1: the full-frame call the pipeline already makes, used as the plan's
    # input. Its own detections are the planning evidence, not the answer.
    seed_tiles, _ = detect_in_tiles(image_path, [(0, 0, fw, fh)], client,
                                    concept=concept, tmp_dir=tmp_dir)
    seed_boxes = seed_tiles[0][1]
    print(f"  seed full-frame call: {len(seed_boxes)} boxes", flush=True)

    plan = plan_tiles(seed_boxes, fw, fh,
                      max_share_after_split=max_share_after_split,
                      max_depth=max_depth)
    print(f"  plan: {plan['stats']['n_tiles']} tiles, depth "
          f"{plan['stats']['max_depth_used']}, stops {plan['stats']['stop_reasons']}",
          flush=True)

    per_tile, fails_a = detect_in_tiles(image_path, plan["tiles"], client,
                                        concept=concept, tmp_dir=tmp_dir)
    adaptive, drop_a = merge_tile_detections(per_tile)
    calls_a = 1 + len(plan["tiles"])        # the seed call counts against it
    print(f"  ADAPTIVE: {len(adaptive)} people in {calls_a} calls "
          f"({len(adaptive)/calls_a:.1f} per call), {drop_a} dupes, "
          f"{len(fails_a)} failures", flush=True)

    uni_tiles = uniform_grid(fw, fh, uniform_n)
    per_tile_u, fails_u = detect_in_tiles(image_path, uni_tiles, client,
                                          concept=concept, tmp_dir=tmp_dir)
    uniform, drop_u = merge_tile_detections(per_tile_u)
    print(f"  UNIFORM {uniform_n}x{uniform_n}: {len(uniform)} people in "
          f"{len(uni_tiles)} calls ({len(uniform)/len(uni_tiles):.1f} per call), "
          f"{drop_u} dupes, {len(fails_u)} failures", flush=True)

    result = {
        "image": str(image_path), "frame": [fw, fh], "concept": concept,
        "seed_boxes": len(seed_boxes),
        "plan_stats": plan["stats"],
        "adaptive": {"n_people": len(adaptive), "n_calls": calls_a,
                     "per_call": round(len(adaptive) / calls_a, 2),
                     "duplicates_dropped": drop_a,
                     "tile_failures": len(fails_a)},
        "uniform": {"grid": f"{uniform_n}x{uniform_n}",
                    "n_people": len(uniform), "n_calls": len(uni_tiles),
                    "per_call": round(len(uniform) / len(uni_tiles), 2),
                    "duplicates_dropped": drop_u,
                    "tile_failures": len(fails_u)},
        "spent_usd": round(client.spent_usd, 4),
    }
    path = out / (pathlib.Path(image_path).stem + "_adaptive_vs_uniform.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  written: {path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="adaptive tiling")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", metavar="IMAGE",
                    help="measure detections per grid level on one frame (costs money)")
    ap.add_argument("--compare", metavar="IMAGE",
                    help="adaptive vs uniform grid on one frame (costs money)")
    ap.add_argument("--refine", metavar="IMAGE",
                    help="progressive yield-driven refinement on one frame (costs money)")
    ap.add_argument("--yield-gain", type=float, default=DEFAULT_YIELD_GAIN)
    ap.add_argument("--uniform-n", type=int, default=3)
    ap.add_argument("--max-share", type=float,
                    default=MAX_PERSON_SHARE_AFTER_SPLIT)
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--concept", default="person")
    ap.add_argument("--max-usd", type=float, default=0.50)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.refine:
        from proxemix.sam_video import SamVideoClient
        client = SamVideoClient(max_usd=args.max_usd)
        if not client.key_present():
            print("no API key — refinement needs one", file=sys.stderr)
            return 1
        print(f"refining {args.refine} (max depth {args.max_depth}, "
              f"gain threshold {args.yield_gain})")
        res = refine_by_yield(args.refine, client, concept=args.concept,
                              max_depth=args.max_depth,
                              yield_gain=args.yield_gain)
        out = pathlib.Path("outputs/tiling_calibration")
        out.mkdir(parents=True, exist_ok=True)
        path = out / (pathlib.Path(args.refine).stem + "_refined.json")
        summary = {k: v for k, v in res.items() if k != "boxes"}
        summary["height_p05"] = round(
            percentile([b[3] - b[1] for b in res["boxes"]], 0.05), 1)
        summary["height_median"] = round(
            percentile([b[3] - b[1] for b in res["boxes"]], 0.50), 1)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  TOTAL: {res['n_people']} people in {res['n_calls']} calls "
              f"({res['n_people']/res['n_calls']:.1f} per call), "
              f"{len(res['tiles'])} tiles in the reusable plan, "
              f"${res['spent_usd']:.4f}")
        print(f"  written: {path}")
        return 0
    if args.compare:
        from proxemix.sam_video import SamVideoClient
        client = SamVideoClient(max_usd=args.max_usd)
        if not client.key_present():
            print("no API key — the comparison needs one", file=sys.stderr)
            return 1
        print(f"comparing on {args.compare}")
        compare_adaptive(args.compare, client, uniform_n=args.uniform_n,
                         concept=args.concept,
                         max_share_after_split=args.max_share,
                         max_depth=args.max_depth)
        return 0
    if args.calibrate:
        from proxemix.sam_video import SamVideoClient
        client = SamVideoClient(max_usd=args.max_usd)
        if not client.key_present():
            print("no API key — calibration needs one", file=sys.stderr)
            return 1
        levels = tuple(int(x) for x in args.levels.split(","))
        print(f"calibrating on {args.calibrate} at levels {levels}")
        calibrate(args.calibrate, client, levels=levels, concept=args.concept)
        return 0
    ap.error("give --selftest or --calibrate")


if __name__ == "__main__":
    sys.exit(main())
