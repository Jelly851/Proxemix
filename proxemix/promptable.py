"""Stage 2b — promptable search: a typed concept becomes frame-cited hits.

Why this module exists
----------------------
`sam_video.py` already takes any text concept — `segment_image(concept=...)`
passes the string straight to SAM, and "person", "face" and "head" are all
proven live. But the pipeline hard-codes person and face, and the chat panel is
a keyword matcher over seven precomputed categories. So "promptable" was true of
the model underneath and not of the product. This module closes that: a concept
typed in the UI runs a real SAM pass over the clip's own frames and comes back
as spans the player can jump to.

The one design decision, and the reason for it
----------------------------------------------
**Search samples frames through the image endpoint; it does not run the video
endpoint.** Three reasons, in the order they decided it:

1. *Search wants coverage, not continuity.* "Where does a wheelchair appear"
   needs one hit per appearance, not an unbroken track. Sampling every Kth frame
   answers the question; tracking is the bulk pass's job.
2. *The video path is not proven live for a local file.* `segment_video`'s own
   docstring says so — whether the hosted API accepts a base64 `video_url` was
   never verified. The image endpoint is the shape proven on 2026-10-27.
3. *The bill is small either way at this length.* The image endpoint is 12.5x
   per frame ($2.50 vs $0.20 per 1,000), but a video request is billed a
   50-frame minimum, so on an 18-frame sample the video endpoint's floor
   ($0.010) and the image sample ($0.045) are both rounding error. The 12.5x
   argument that shaped the bulk pass does not reach down here.

What a run returns, and what it refuses to imply
------------------------------------------------
Every hit carries the canonical frame, the box, SAM's confidence, and a citation
rendered at the bundle's own fps. A concept SAM did not find comes back
`found: False` **with the sampling named** — frames sampled out of frames total,
and the stride. Absence from a sample is not absence from the clip, and this
module never prints it as though it were.

Repeat questions are free: results are cached per (concept, stride) under
`<bundle>/searches/`, so a demo can ask the same thing twice without paying
twice. `--refresh` forces a new pass.

Run it
------
    py -m proxemix.promptable --selftest                       # mocked, no key
    py -m proxemix.promptable --bundle data/plaza_bundle --concept "backpack"
    py -m proxemix.promptable --bundle data/plaza_bundle --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

from .sam_video import (
    IMAGE_USD_PER_1K,
    FrameSet,
    SamError,
    SamVideoClient,
    detections_to_frame_table,
    image_cost_usd,
    parse_segmentation,
)
from .timecode import frame_citation

# Frames are extracted once per bundle and reused by every later search, so the
# second concept somebody types costs nothing but the API calls.
SEARCH_FRAMES_DIR = "search_frames"
SEARCH_CACHE_DIR = "searches"

DEFAULT_STRIDE = 3          # every 3rd frame — 0.6 s apart on a 5 fps bundle
DEFAULT_MAX_FRAMES = 40     # the ceiling a UI request may spend without asking
DEFAULT_MIN_CONF = 0.0      # SAM's own threshold is the gate; this is a second one

# A concept is a free-text string from a text box, and it ends up in a filename.
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class SearchError(RuntimeError):
    """A search could not be run. Never carries a key or a path to one."""


def slugify(concept):
    """`"yellow safety vest"` -> `"yellow-safety-vest"`, for the cache filename."""
    slug = _SLUG_STRIP.sub("-", str(concept).strip().lower()).strip("-")
    return slug[:60] or "concept"


def estimate_search_usd(n_sampled):
    """What a search over `n_sampled` frames costs, before anything is sent."""
    return image_cost_usd(int(n_sampled))


def bundle_meta(bundle_dir):
    path = pathlib.Path(bundle_dir) / "bundle_meta.json"
    if not path.exists():
        raise SearchError(f"{path} not found — that is not a built bundle")
    return json.loads(path.read_text(encoding="utf-8"))


def bundle_fps(bundle_dir, meta=None):
    meta = meta or bundle_meta(bundle_dir)
    try:
        fps = float(meta.get("fps") or 0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0:
        raise SearchError(
            f"bundle {bundle_dir} declares fps {meta.get('fps')!r} — a citation "
            "cannot be rendered without it")
    return fps


def ensure_search_frames(bundle_dir, fps=None, meta=None):
    """Extract the bundle's video into frames once, and reuse them after.

    Extraction runs at the bundle's **own** fps, which is what makes the
    extracted frame index equal the canonical frame number the tracks table
    uses. Extracting at any other rate would put every hit on the wrong moment —
    the same silent failure `sam_video.build_clip` counts frames to avoid.
    """
    bundle_dir = pathlib.Path(bundle_dir)
    meta = meta or bundle_meta(bundle_dir)
    fps = fps or bundle_fps(bundle_dir, meta)
    frames_dir = bundle_dir / SEARCH_FRAMES_DIR
    manifest = frames_dir / "manifest.json"

    if manifest.exists():
        try:
            man = json.loads(manifest.read_text(encoding="utf-8"))
            if abs(float(man.get("fps", 0)) - fps) < 1e-6 and man.get("frame_count"):
                return FrameSet.from_dir(frames_dir)
        except (ValueError, OSError, TypeError):
            pass    # a broken cache is re-extracted rather than trusted

    video = bundle_dir / meta.get("video_file", "video.mp4")
    if not video.exists():
        raise SearchError(f"{video} not found — nothing to search")

    from .ingest import extract_frames     # imported here: av is heavy
    extract_frames(video, frames_dir, fps=fps)
    return FrameSet.from_dir(frames_dir)


def sample_frames(frame_set, stride=DEFAULT_STRIDE, max_frames=DEFAULT_MAX_FRAMES):
    """Positions in `frame_set` a search will actually send, and why that many.

    Stride first, then the ceiling: a stride of 3 over 90 frames gives 30
    samples, and `max_frames` then thins it evenly rather than truncating it, so
    a capped search still covers the whole clip instead of its first half.
    """
    stride = max(1, int(stride))
    positions = list(range(0, len(frame_set), stride))
    if max_frames and len(positions) > int(max_frames):
        n = int(max_frames)
        step = (len(positions) - 1) / float(n - 1) if n > 1 else 1.0
        picked = sorted({int(round(i * step)) for i in range(n)})
        positions = [positions[i] for i in picked if i < len(positions)]
    return positions


def _spans_from_frames(hit_positions, frame_set, counts, fps):
    """Consecutive sampled hits -> one cited span each.

    Adjacency is measured in *sampled* positions, not canonical frames: two hits
    on neighbouring samples are one appearance, and a sample with nothing in it
    between them ends the span. That is the honest reading of a strided pass —
    it cannot claim continuity across a frame it never looked at.
    """
    spans = []
    run = []
    prev_idx = None
    for idx, pos in hit_positions:
        if prev_idx is not None and idx != prev_idx + 1:
            spans.append(run)
            run = []
        run.append((idx, pos))
        prev_idx = idx
    if run:
        spans.append(run)

    out = []
    for run in spans:
        frames = [frame_set.canonical(pos) for _, pos in run]
        peak_pos = max(run, key=lambda ip: counts.get(ip[1], 0))[1]
        out.append({
            "frame_start": int(min(frames)),
            "frame_end": int(max(frames)),
            "citation": frame_citation(min(frames), max(frames), fps=fps),
            "n_samples": len(run),
            "peak_count": int(max(counts.get(pos, 0) for _, pos in run)),
            "peak_frame": int(frame_set.canonical(peak_pos)),
        })
    return out


def search(bundle_dir, concept, stride=DEFAULT_STRIDE,
           max_frames=DEFAULT_MAX_FRAMES, client=None, use_cache=True,
           min_conf=DEFAULT_MIN_CONF, max_usd=None):
    """Run one concept over a bundle's frames and return frame-cited hits.

    Raises `SearchError` on anything that would make the result dishonest — a
    missing bundle, an unknown fps, a cost above `max_usd`. A per-frame API
    failure is **recorded and survived**: one bad frame out of thirty should not
    lose the pass, and the result says how many frames failed so a thin answer
    can be read as thin.
    """
    bundle_dir = pathlib.Path(bundle_dir)
    concept = str(concept).strip()
    if not concept:
        raise SearchError("a search needs a concept")

    meta = bundle_meta(bundle_dir)
    fps = bundle_fps(bundle_dir, meta)
    slug = slugify(concept)

    if use_cache:
        cached = load_cached(bundle_dir, concept, stride)
        if cached is not None:
            return {**cached, "cached": True}

    frame_set = ensure_search_frames(bundle_dir, fps=fps, meta=meta)
    positions = sample_frames(frame_set, stride=stride, max_frames=max_frames)
    if not positions:
        raise SearchError(f"bundle {bundle_dir} yielded no frames to sample")

    estimate = estimate_search_usd(len(positions))
    cap = float(max_usd) if max_usd is not None else round(estimate * 1.5 + 0.01, 4)
    if max_usd is not None and estimate > float(max_usd) + 1e-9:
        raise SearchError(
            f"searching {len(positions)} frames for {concept!r} would cost "
            f"${estimate:.4f}, above the ${float(max_usd):.4f} cap — raise the "
            "cap or the stride")

    client = client or SamVideoClient(max_usd=cap)

    detections, counts, failures = [], {}, []
    for pos in positions:
        path = frame_set.paths[pos]
        try:
            text = client.segment_image(path, concept=concept)
        except SamError as e:
            failures.append({"frame": int(frame_set.canonical(pos)),
                             "error": str(e)[:300]})
            continue
        parsed = parse_segmentation(text)
        # One image is one frame: SAM's own `<Nf>` inside a single-image
        # response is 0 (or absent), so the clip index is this sample's
        # position, never the token's number.
        flat = []
        for dets in parsed.values():
            flat.extend(dets)
        table = detections_to_frame_table({pos: flat}, frame_set=frame_set,
                                          concept=concept)
        kept = 0
        for row in table.to_dict(orient="records"):
            conf = row.get("conf")
            if min_conf and conf is not None and float(conf) < float(min_conf):
                continue
            frame = int(row["frame"])
            detections.append({
                "frame": frame,
                "t": round(frame / fps, 3),
                "object_id": int(row["object_id"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
                "conf": None if conf is None else round(float(conf), 4),
                "citation": frame_citation(frame, frame, fps=fps),
            })
            kept += 1
        counts[pos] = kept

    hit_positions = [(i, pos) for i, pos in enumerate(positions)
                     if counts.get(pos, 0) > 0]
    spans = _spans_from_frames(hit_positions, frame_set, counts, fps)
    sampled_frames = [int(frame_set.canonical(p)) for p in positions]

    result = {
        "concept": concept,
        "slug": slug,
        "stride": int(max(1, stride)),
        "fps": fps,
        "n_frames_total": len(frame_set),
        "n_frames_sampled": len(positions),
        "frames_sampled": sampled_frames,
        "found": bool(detections),
        "n_detections": len(detections),
        "peak_count": int(max(counts.values())) if counts else 0,
        "frames_with_hits": len(hit_positions),
        "spans": spans,
        "detections": detections,
        "failed_frames": failures,
        "cost_usd": round(client.spent_usd, 4),
        "estimate_usd": round(estimate, 4),
        "coverage": _coverage_note(len(positions), len(frame_set),
                                   max(1, int(stride)), fps, bool(detections)),
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cached": False,
    }
    _write_cache(bundle_dir, result)
    return result


def _coverage_note(n_sampled, n_total, stride, fps, found):
    """The sentence that keeps a null result from reading as proof of absence."""
    gap = stride / float(fps)
    base = (f"Sampled {n_sampled} of {n_total} frames — every {stride}"
            f"{'st' if stride == 1 else 'rd' if stride == 3 else 'nd' if stride == 2 else 'th'}"
            f" frame, {gap:.2f}s apart.")
    if found:
        return base + (" Counts are per sampled frame, so an appearance shorter "
                       f"than {gap:.2f}s may sit between two samples.")
    return base + (" Nothing was found on those frames. That is not the same as "
                   "absence from the clip: lower the stride to look harder.")


# --- cache -----------------------------------------------------------------

def _cache_path(bundle_dir, concept, stride):
    return (pathlib.Path(bundle_dir) / SEARCH_CACHE_DIR
            / f"{slugify(concept)}-s{max(1, int(stride))}.json")


def _write_cache(bundle_dir, result):
    path = _cache_path(bundle_dir, result["concept"], result["stride"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def load_cached(bundle_dir, concept, stride=DEFAULT_STRIDE):
    path = _cache_path(bundle_dir, concept, stride)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def list_searches(bundle_dir):
    """Every concept already run against this bundle, newest first."""
    d = pathlib.Path(bundle_dir) / SEARCH_CACHE_DIR
    if not d.is_dir():
        return []
    out = []
    for path in d.glob("*.json"):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        out.append({
            "concept": r.get("concept"),
            "stride": r.get("stride"),
            "found": r.get("found"),
            "n_detections": r.get("n_detections"),
            "frames_with_hits": r.get("frames_with_hits"),
            "cost_usd": r.get("cost_usd"),
            "ran_at": r.get("ran_at"),
        })
    return sorted(out, key=lambda r: r.get("ran_at") or "", reverse=True)


def render(result):
    """A search as somebody reading a terminal would want it."""
    w = 74
    head = "FOUND" if result["found"] else "NOT FOUND"
    out = [
        "=" * w,
        f"  {result['concept']!r} — {head}",
        "=" * w,
        f"  {result['n_detections']} detection(s) across "
        f"{result['frames_with_hits']} of {result['n_frames_sampled']} sampled "
        f"frames; peak {result['peak_count']} in one frame",
        f"  cost ${result['cost_usd']:.4f}"
        + ("  (cached — nothing was sent)" if result.get("cached") else ""),
        "-" * w,
    ]
    for s in result["spans"]:
        out.append(f"  {s['citation']}   peak {s['peak_count']} "
                   f"(frame {s['peak_frame']:,})")
    if result.get("failed_frames"):
        out.append(f"  {len(result['failed_frames'])} frame(s) failed and were "
                   "skipped")
    out += ["-" * w, f"  {result['coverage']}", "=" * w]
    return "\n".join(out)


# --- self-test -------------------------------------------------------------

def _mock_transport(hits_on):
    """A transport that answers as SAM does, with no key and no network.

    `hits_on` is the set of 1-based call numbers that return a box, which is how
    the span grouping is tested: a gap in the middle must become two spans.
    """
    state = {"n": 0}

    def transport(url, headers, body, timeout):
        state["n"] += 1
        if state["n"] in hits_on:
            payload = ("<0f> 1 <|box;x1=10;y1=20;x2=60;y2=140;w=2560;h=1440;c=0.91|>"
                       " 2 <|box;x1=300;y1=40;x2=360;y2=170;w=2560;h=1440;c=0.78|>")
        else:
            payload = "<0f>"
        return 200, f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': payload})}\n"

    return transport


def selftest():
    """Proves the grouping and the honesty, with no key and no API."""
    import tempfile

    from PIL import Image

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        bundle = pathlib.Path(tmp) / "bundle"
        frames = bundle / SEARCH_FRAMES_DIR
        frames.mkdir(parents=True)
        for i in range(9):
            Image.new("RGB", (64, 36), (20, 20, 20)).save(
                frames / f"frame_{i:06d}.jpg", quality=60)
        (frames / "manifest.json").write_text(json.dumps({
            "fps": 5.0, "frame_count": 9,
            "frames": [{"index": i, "path": str(frames / f'frame_{i:06d}.jpg')}
                       for i in range(9)]}), encoding="utf-8")
        (bundle / "bundle_meta.json").write_text(json.dumps({
            "fps": 5.0, "video_file": "video.mp4"}), encoding="utf-8")

        # stride 1 over 9 frames: hits on samples 1,2 and 5 -> two spans
        client = SamVideoClient(key_path=frames / "manifest.json",
                                transport=_mock_transport({1, 2, 5}), max_usd=1.0)
        r = search(bundle, "orange cone", stride=1, client=client, use_cache=False)

        checks = [
            ("found", r["found"] is True),
            ("9 frames sampled", r["n_frames_sampled"] == 9),
            ("6 detections", r["n_detections"] == 6),
            ("two spans", len(r["spans"]) == 2),
            ("first span is frames 0-1", (r["spans"][0]["frame_start"],
                                          r["spans"][0]["frame_end"]) == (0, 1)),
            ("second span is frame 4", (r["spans"][1]["frame_start"],
                                        r["spans"][1]["frame_end"]) == (4, 4)),
            ("citation at bundle fps", "00:00" in r["spans"][0]["citation"]),
            ("cost counted", r["cost_usd"] > 0),
            ("cache written", load_cached(bundle, "orange cone", 1) is not None),
        ]

        # a concept with no hits must say so, and must not claim absence
        client2 = SamVideoClient(key_path=frames / "manifest.json",
                                 transport=_mock_transport(set()), max_usd=1.0)
        r2 = search(bundle, "wheelchair", stride=3, client=client2, use_cache=False)
        checks += [
            ("miss reports not found", r2["found"] is False),
            ("miss has no spans", r2["spans"] == []),
            ("miss qualifies absence", "not the same as absence" in r2["coverage"]),
            ("listed two searches", len(list_searches(bundle)) == 2),
        ]

        for name, passed in checks:
            print(f"  {'ok  ' if passed else 'FAIL'} {name}")
            ok = ok and passed

        print("\n" + render(r))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Promptable concept search over a bundle.")
    ap.add_argument("--bundle", default="data/plaza_bundle")
    ap.add_argument("--concept")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--max-usd", type=float, default=None,
                    help="refuse the search if the estimate exceeds this")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--list", action="store_true", help="concepts already run")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.list:
        rows = list_searches(args.bundle)
        if not rows:
            print(f"no searches cached under {args.bundle}")
            return 0
        for r in rows:
            print(f"  {r['concept']!r:<28} stride {r['stride']}  "
                  f"{r['n_detections']:>4} det  ${r['cost_usd']:.4f}  {r['ran_at']}")
        return 0

    if not args.concept:
        ap.error("give --concept, --list or --selftest")

    try:
        result = search(args.bundle, args.concept, stride=args.stride,
                        max_frames=args.max_frames, max_usd=args.max_usd,
                        use_cache=not args.refresh)
    except SearchError as e:
        print(f"{e}", file=sys.stderr)
        return 1
    print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
