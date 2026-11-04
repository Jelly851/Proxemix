"""Does tiling help when people are actually small? Measured against ground truth.

The problem this exists to solve. Tiling's recall gain was measured two ways
and neither was conclusive:

* **On plaza footage** detections went 65 -> 103 -> 118 -> 167 as the frame was
  cut finer, and plaza is where people *are* small — median 112 px in a 1440
  frame, smallest 5% at 28 px. But plaza has no ground truth, so every one of
  those numbers is self-graded.
* **On SALSA** there is ground truth, and tiling moved recall 64.6% -> 74.5%
  while precision fell 65.3% -> 29.8%. But SALSA's people are a median 195 px
  tall in a 768-tall frame — **25% of the frame height**, nowhere near the
  detection floor. Its full-frame pass already returns 17 boxes per frame
  against 17 annotated people. SALSA's missing third is occlusion and camera
  coverage, which no amount of cutting can recover.

So the two datasets each answer half the question. This module makes them
answer all of it, by **downscaling the SALSA frames**: the same annotated
people, the same homography, the same scoring, with person size as the only
thing that changes. At 1/3 scale a 195 px participant becomes 65 px, which is
where the plaza calibration put the detection floor.

If tiling earns its cost anywhere, it earns it here, and here the answer is
checkable.

    py -m proxemix.tiling_scale_eval --scale 0.33 --frames 12
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

from proxemix import sam_video
from proxemix.salsa_adapter import load_salsa_tracks
from proxemix.salsa_camera import SalsaCamera
from proxemix.salsa_sam_facing import (
    DEFAULT_FRAMES_DIR,
    DEFAULT_PART,
    DEFAULT_SALSA_ROOT,
    list_frames,
)
from proxemix.tiling import (
    detect_in_tiles,
    percentile,
    plan_tiles,
    uniform_grid,
)
from proxemix.tiling_sweep import evaluate, merge_by_containment

OUT_DIR = pathlib.Path("outputs/tiling_calibration")


def downscale(path, scale, out_dir):
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.LANCZOS)
    out = os.path.join(out_dir, os.path.basename(path))
    small.save(out, quality=95)
    return out, small.size


def upscale_boxes(boxes, scale):
    """Boxes detected on the shrunk frame, expressed in original coordinates.

    Scoring happens in the original frame's pixel space because that is the
    space the camera calibration and the annotations live in. Detecting small
    and scoring large is the whole trick, and forgetting this step would put
    every detection at a third of its real position.
    """
    return [[b[0] / scale, b[1] / scale, b[2] / scale, b[3] / scale,
             b[4] if len(b) > 4 else 0.0] for b in boxes]


def run(scale=0.33, n_frames=12, containment=0.7, conf_min=0.0,
        frames_dir=DEFAULT_FRAMES_DIR, part=DEFAULT_PART, max_usd=3.0,
        grid=2):
    from PIL import Image

    cam = SalsaCamera.cam(1, os.path.join(DEFAULT_SALSA_ROOT, "calibration"))
    gt = load_salsa_tracks(DEFAULT_SALSA_ROOT, part)
    available = list_frames(frames_dir)
    if not available:
        print(f"no frames under {frames_dir}", file=sys.stderr)
        return 1
    step = max(1, len(available) // n_frames)
    picked = available[::step][:n_frames]

    client = sam_video.SamVideoClient(max_usd=max_usd)
    if not client.key_present():
        print("no API key", file=sys.stderr)
        return 1

    tmp = tempfile.mkdtemp(prefix="proxemix_scale_")
    orig_w, orig_h = Image.open(picked[0][1]).size
    small_paths = [(canon, downscale(p, scale, tmp)) for canon, p in picked]
    sw, sh = small_paths[0][1][1]
    print(f"frames {orig_w}x{orig_h} -> {sw}x{sh} at scale {scale}")

    # --- arm 1: full shrunk frame, one call each -----------------------------
    base_rows = []
    for canon, (sp, _) in small_paths:
        per_tile, _ = detect_in_tiles(sp, [(0, 0, sw, sh)], client, tmp_dir=tmp)
        base_rows.extend((canon, b) for b in upscale_boxes(per_tile[0][1], scale))
    base = evaluate(base_rows, gt, cam)
    base["n_calls"] = len(picked)
    heights = [b[3] - b[1] for _, b in base_rows]
    base["height_median_original_px"] = round(percentile(heights, 0.5), 1)
    base["height_median_as_sent_px"] = round(percentile(heights, 0.5) * scale, 1)
    print(f"  FULL FRAME: recall {base['recall']} prec {base['precision']} "
          f"F1 {base['f1']} ({base['boxes']} boxes, {base['n_calls']} calls, "
          f"people {base['height_median_as_sent_px']}px as sent)", flush=True)

    # --- does the share rule even want to tile this? -------------------------
    seed = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
            for _, b in base_rows[:400]]
    plan = plan_tiles(seed, sw, sh)
    print(f"  share rule on the shrunk frame: {plan['stats']['n_tiles']} tiles, "
          f"p25 person {plan['stats']['person_height_p25']}px of {sh}px "
          f"({plan['stats']['person_height_p25']/sh:.1%})", flush=True)

    # --- arm 2: uniform grid on the shrunk frame -----------------------------
    tiles = uniform_grid(sw, sh, grid)
    tiled_rows = []
    for i, (canon, (sp, _)) in enumerate(small_paths, 1):
        per_tile, _ = detect_in_tiles(sp, tiles, client, tmp_dir=tmp)
        flat = []
        for _, boxes in per_tile:
            flat.extend(boxes)
        merged = merge_by_containment(flat, iou_threshold=0.45,
                                      containment=containment)
        tiled_rows.extend((canon, b) for b in upscale_boxes(merged, scale))
        if i % 4 == 0:
            print(f"    {i}/{len(picked)} frames, ${client.spent_usd:.3f}",
                  flush=True)
    tiled = evaluate(tiled_rows, gt, cam)
    tiled["n_calls"] = len(picked) * len(tiles)
    print(f"  {grid}x{grid} TILED: recall {tiled['recall']} "
          f"prec {tiled['precision']} F1 {tiled['f1']} "
          f"({tiled['boxes']} boxes, {tiled['n_calls']} calls)", flush=True)

    result = {
        "scale": scale, "grid": f"{grid}x{grid}", "n_frames": len(picked),
        "original_frame": [orig_w, orig_h], "sent_frame": [sw, sh],
        "containment": containment, "conf_min": conf_min,
        "full_frame": base, "tiled": tiled,
        "recall_gain_pp": round(100 * (tiled["recall"] - base["recall"]), 1),
        "f1_gain": round(tiled["f1"] - base["f1"], 4),
        "cost_multiple": round(tiled["n_calls"] / max(1, base["n_calls"]), 1),
        "spent_usd": round(client.spent_usd, 4),
    }
    verdict = ("TILING WINS — recall and F1 both up"
               if result["f1_gain"] > 0 and result["recall_gain_pp"] > 0
               else "tiling does not win on F1 even with small people")
    result["verdict"] = verdict
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"salsa_scale{str(scale).replace('.', '')}_{grid}x{grid}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{verdict}")
    print(f"  recall {base['recall']} -> {tiled['recall']} "
          f"({result['recall_gain_pp']:+} points), "
          f"F1 {base['f1']} -> {tiled['f1']} ({result['f1_gain']:+}), "
          f"{result['cost_multiple']}x calls, ${result['spent_usd']:.3f}")
    print(f"written: {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="tiling vs person size, with truth")
    ap.add_argument("--scale", type=float, default=0.33)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--containment", type=float, default=0.7)
    ap.add_argument("--max-usd", type=float, default=3.0)
    args = ap.parse_args()
    return run(scale=args.scale, n_frames=args.frames, grid=args.grid,
               containment=args.containment, max_usd=args.max_usd)


if __name__ == "__main__":
    sys.exit(main())
