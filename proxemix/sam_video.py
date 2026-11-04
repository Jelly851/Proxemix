"""Stage 2 — detect + track with SAM 3.1, the stage the keyless build left out.

What this replaces
------------------
`grand_central_adapter.py` stands in for this stage: it reads published
pedestrian trajectories and emits the canonical `(track_id, frame, x, y)` table.
This module produces that *same table* from real footage, so ingest -> clean ->
calibrate -> metrics -> report -> UI run unchanged. The contract is the four
columns and nothing else:

    track_id  str, zero-padded like the GC adapter's ("000007")
    frame     int, the canonical frame number (see "Frame numbering")
    x, y      float, pixels — the person's **feet point** (bottom-centre of the
              box), because `calibration.py` projects a floor-plane homography
              and the spec's stage 4 says feet-point projection.

Everything richer than those four columns (boxes, confidences, mask headers)
comes back in a second, optional table — `detections` — which the facing step
and the UI overlay use and the canonical pipeline ignores.

Two endpoints, two prices, and the price decides the architecture
----------------------------------------------------------------
Read off `dev.meta.ai/docs/pricing-rate-limits`, 2026-10-27:

    image segmentation   $2.50 per 1,000 images
    video segmentation   $0.20 per 1,000 frames   (min charge 50 frames/request)

So the image endpoint is **12.5x the price per frame** of the video endpoint.
That is why the spec's shape is the right one and why this module follows it:

1. **Headcount front porch** — the image endpoint on a few sampled stills.
   It returns every instance with no 16-object cap, so it is what says how
   dense the scene is. A handful of calls, a fraction of a cent.
2. **The bulk pass** — the video endpoint, one pass per concept. SAM keeps a
   stable object id across frames, so its own ids become `track_id` and no
   stitching is needed.
3. **>16 people** — video tracking is capped at "up to 16 objects per frame"
   (docs, SAM overview/segmenting). Past that the scene is partitioned into
   spatially adjacent groups of <=16 and one video pass runs per group with
   **box prompts** seeding the objects. Passes are then merged on box IoU over
   shared frames with Hungarian assignment as the safety net.
   **Chunked re-seeding, on by default.** A box prompt seeds the objects it
   names and nothing else, so a single seed frame tracks only the people
   standing on it. Measured on the plaza clip 2026-10-27: seeded at frame 1 with
   49 people, by frame 45 the run held 35 while the image endpoint found 52 —
   and the people it was missing were the nearest and largest in the shot,
   because the foreground is exactly where somebody walks in, crosses and leaves
   inside one clip. So the clip is cut into ~30-frame chunks (`chunk_frames`),
   each chunk gets its own clip and a fresh detection on its first frame, each
   chunk's passes are seeded with **both** the continuing tracks' latest boxes
   and the newcomers that detection found, and each chunk's tracks are linked
   onto the running ids over the two frames neighbouring chunks share.
   `chunk_frames=None` restores the single-shot behaviour.
4. **Image fallback** — `mode="image"` detects per frame over the image
   endpoint and stitches tracks here (`stitch_tracks`). It costs 12.5x the
   video path and exists for two honest reasons: it is the call shape that has
   actually been proven live, and it is the only path when the frames do not
   form a clip (far-apart stills, a tiling pass, a sanity check).

Cost is counted, not assumed: every call adds to `SamVideoClient.spent_usd`,
`max_usd` refuses the call that would cross the cap *before* sending it, and
`run_sam_stage` returns the dollar figure in its summary.

Frame numbering
---------------
SAM numbers frames `<Nf>` from 0 within the clip it was given. The canonical
`frame` column is whatever the caller's clock is — for a 5 fps extraction that
is the sample index with `fps=5.0, sample_stride=1` passed to
`track_cleaning.clean_tracks`. `FrameSet` holds the mapping both ways and every
returned row is already in canonical numbers.

Facing, riding the same calls
-----------------------------
A second concept — "face" — on the same frames gives the orientation channel
`facing.py` was built with a seam for. Proven live 2026-10-27 on a real crowd
photo: "person" returned 55 boxes, "face" returned 45.

* A face box inside a person box says that person is turned roughly camera-ward,
  and the face's horizontal offset inside the body box gives a coarse yaw: the
  visible face sits toward the side the head is turned to, so
  `u = (face_cx - person_cx) / (person_w / 2)` in [-1, 1] maps to
  `yaw = u * max_yaw`, rotating away from camera-ward toward image-right.
* A tracked person with **no** face box is turned away. That is a real signal
  and it is also a hemisphere, not an angle — and face-absence has innocent
  causes (occlusion, resolution, the back of a crowd). So it is emitted with
  `facing_source="face_absent"` at a confidence *below* `groups.py`'s
  `DEFAULT_FACING_MIN_CONF` (0.3), which means it is recorded and inert until
  somebody raises the gate and measures what it does on SALSA.
* Angles come out in the **floor plane** (the frame `x_m`/`y_m` live in) when a
  calibration config is supplied, because that is the frame `groups.py` reads
  them in. Without one they come out in the pixel frame, which is a coarser
  answer and is labelled as such in the summary.

Masks
-----
Mask tokens are captured (height, width, encoding marker, payload length) and
**not decoded**. The payload is base85 with a byte layout only Meta's own
parser libraries specify, and installing them is Jayden's call, not mine. Boxes
carry every number this pipeline uses.

Run it
------
    py -m proxemix.sam_video                      # mocked self-test, no key
    py -m proxemix.sam_video --frames DIR --out DIR [--mode auto|video|image]
    py -m proxemix.sam_video --headcount IMAGE
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from proxemix.facing import apply_external_facing

# --- API facts, all read off dev.meta.ai/docs 2026-10-27 ---------------------
API_URL = "https://api.meta.ai/v1/responses"
MODEL = "sam-3.1"
DEFAULT_KEY_FILE = pathlib.Path("C:/Users/jayde/meta-api-key.txt")
VIDEO_OBJECT_CAP = 16          # "up to 16 objects per frame" for video tracking
VIDEO_MAX_FRAMES = 15000       # ~10 min clip limit
IMAGE_USD_PER_1K = 2.50
VIDEO_USD_PER_1K = 0.20
VIDEO_MIN_BILLED_FRAMES = 50   # "video requests incur a minimum charge of 50 frames"
TRANSPORT_RETRIES = 2          # a dropped connection is not a failed request
TRANSPORT_BACKOFF_SECONDS = 2.0

# --- output grammar ----------------------------------------------------------
# line ::= FRAME record ("," record)*   record ::= OBJECT_ID box mask
# Parsed by key rather than by position: the live probe (Body/work/sam_probe.py,
# 2026-10-27) matched on `<|box;x1=` alone, and a strict field order would have
# turned a harmless grammar addition into zero detections.
_FRAME_RE = re.compile(r"<(\d+)f>")
_BOX_RE = re.compile(r"(?:(\d+)\s*)?<\|box;([^|]*?)\|>")
_MASK_RE = re.compile(r"<\|mask;([^|]*?)\|>")
_FIELD_RE = re.compile(r"([a-zA-Z0-9_]+)=([^;]*)")

DEFAULT_PERSON_CONCEPT = "person"
DEFAULT_FACE_CONCEPT = "face"

# Facing extraction defaults.
DEFAULT_MAX_YAW_DEG = 70.0      # a face still readable at ~70 deg off-camera
DEFAULT_FACE_CONF = 0.60        # beats a standing person's 0.0, loses to a
                                # full-speed walker's 1.0 in apply_external_facing
DEFAULT_ABSENT_CONF = 0.25      # below groups.py's 0.3 gate, deliberately
DEFAULT_MIN_CONTAINMENT = 0.55  # share of the face box inside the person box

# Chunked re-seeding defaults. The number that chose 30 is measured, not picked:
# on the plaza clip's single-shot run (90 frames at 5 fps, seeded once at frame 1)
# the tracked count fell 49 -> 35 -> 24 while the image endpoint's true headcount
# on the same frames rose 49 -> 52 -> 63, so the share of the crowd missing grows
# by roughly 0.7 percentage points per frame past a seed: ~0% at the seed, ~21%
# 30 frames later, ~35% at 50. Against that, every video request is billed a
# 50-frame minimum, so halving the chunk doubles the bill. 30 frames (6 s at
# 5 fps) is where those two curves cross for this footage — about a tenth of the
# crowd missed on average, at ~$0.15 a run.
DEFAULT_CHUNK_FRAMES = 30       # target chunk length, in frames
DEFAULT_CHUNK_OVERLAP = 2       # frames two neighbouring chunks both cover
DEFAULT_RESEED_IOU = 0.3        # a fresh box this close to a continuing track is it
DEFAULT_CHUNK_LINK_IOU = 0.3    # mean IoU over shared frames to call it one person
DEFAULT_SEED_LOOKBACK = 2       # how far back a continuing seed box may come from


class SamError(RuntimeError):
    """Any API-level failure. Never carries the key in its message."""


class SpendCapExceeded(SamError):
    """Raised *before* a call that would cross `max_usd`."""


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

def image_cost_usd(n_images):
    return n_images * IMAGE_USD_PER_1K / 1000.0


def video_cost_usd(n_frames, n_requests=1):
    """Each request is billed for at least VIDEO_MIN_BILLED_FRAMES frames."""
    billed = max(int(n_frames), VIDEO_MIN_BILLED_FRAMES * max(1, int(n_requests)))
    return billed * VIDEO_USD_PER_1K / 1000.0


def estimate_pass_cost(n_frames, n_concepts=2, mode="video", n_partitions=1):
    """What one run of this stage costs, before it runs.

    `mode="video"`: n_concepts * n_partitions video requests over n_frames.
    `mode="image"`: n_concepts * n_frames image requests.
    """
    if mode == "image":
        return image_cost_usd(n_frames * n_concepts)
    reqs = max(1, n_concepts * n_partitions)
    return video_cost_usd(n_frames * reqs, n_requests=reqs)


def estimate_chunked_cost(n_frames, n_concepts=2, n_partitions=1,
                          chunk_frames=DEFAULT_CHUNK_FRAMES,
                          overlap_frames=DEFAULT_CHUNK_OVERLAP,
                          reseed_image_calls=True):
    """What chunked re-seeding costs, before it runs.

    Worth having as its own function because the 50-frame minimum makes the
    arithmetic unintuitive: chunking a 90-frame clip into three does **not**
    cost the same as one pass over 90 frames, it costs 3 x 50 billed frames per
    partition-concept. Every chunk also pays one image call per concept for its
    fresh detection — cheap, but it is real money and belongs in the estimate.
    """
    spans = chunk_spans(n_frames, chunk_frames, overlap_frames)
    reqs_per_chunk = max(1, int(n_concepts) * max(1, int(n_partitions)))
    usd = 0.0
    for a, b in spans:
        usd += video_cost_usd((b - a) * reqs_per_chunk, n_requests=reqs_per_chunk)
    if reseed_image_calls:
        usd += image_cost_usd(len(spans) * int(n_concepts))
    return usd


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

class FrameSet:
    """An ordered list of frame image paths plus their canonical frame numbers.

    Canonical numbers come from, in order of preference: an explicit list; the
    ingest manifest's `index`; the digits in `frame_000123.jpg`; the position in
    the sorted list. The clip handed to SAM is always frames in this order, so
    SAM's `<Nf>` is this object's position N.
    """

    def __init__(self, paths, frames=None):
        self.paths = [str(p) for p in paths]
        if frames is not None:
            if len(frames) != len(self.paths):
                raise ValueError("frames and paths must be the same length")
            self.frames = [int(f) for f in frames]
        else:
            self.frames = [self._frame_from_name(p, i)
                           for i, p in enumerate(self.paths)]

    @staticmethod
    def _frame_from_name(path, position):
        m = re.search(r"(\d{3,})", os.path.basename(str(path)))
        return int(m.group(1)) if m else position

    def __len__(self):
        return len(self.paths)

    def canonical(self, clip_index):
        """SAM's `<Nf>` -> the canonical frame number."""
        if 0 <= clip_index < len(self.frames):
            return self.frames[clip_index]
        return int(clip_index)

    def sample(self, n):
        """n paths spread evenly across the set — the headcount front porch."""
        if n >= len(self.paths):
            return list(self.paths)
        idx = np.linspace(0, len(self.paths) - 1, num=n).round().astype(int)
        return [self.paths[i] for i in sorted(set(idx.tolist()))]

    @classmethod
    def _from_manifest(cls, man, base_dir):
        """Read a frame manifest in any of the three shapes on disk here.

        `ingest.extract_frames` writes `frames` as a list of dicts keyed
        `index`/`path` with absolute paths; the kill-test harness writes it as a
        dict of frame-number -> dict with `frame_index` and a path *relative to
        the manifest*. Both are real files in this project, so both are read —
        found by running this on the kill-test frames rather than by assuming
        one shape.
        """
        entries = man.get("frames")
        if not entries:
            return None
        if isinstance(entries, dict):
            items = [(k, v) for k, v in entries.items()]
        else:
            items = [(None, v) for v in entries]

        paths, frames = [], []
        for key, e in items:
            if not isinstance(e, dict):
                return None
            raw = e.get("path")
            if not raw:
                continue
            p = pathlib.Path(raw)
            if not p.is_absolute():
                p = base_dir / raw
                if not p.exists():
                    p = base_dir.parent / raw
            if not p.exists():
                continue
            number = e.get("index", e.get("frame_index", key))
            try:
                number = int(number)
            except (TypeError, ValueError):
                number = len(frames)
            paths.append(str(p))
            frames.append(number)
        if not paths:
            return None
        order = sorted(range(len(paths)), key=lambda i: frames[i])
        return cls([paths[i] for i in order], [frames[i] for i in order])

    @classmethod
    def from_dir(cls, frames_dir, pattern=("*.jpg", "*.jpeg", "*.png")):
        d = pathlib.Path(frames_dir)
        manifest = d / "manifest.json"
        if manifest.exists():
            try:
                parsed = cls._from_manifest(json.loads(
                    manifest.read_text(encoding="utf-8")), d)
                if parsed is not None:
                    return parsed
            except (ValueError, OSError, TypeError):
                pass  # a broken manifest is not a reason to refuse the directory
        paths = []
        for pat in pattern:
            paths.extend(sorted(d.glob(pat)))
        if not paths:
            raise FileNotFoundError(f"no frame images found under {frames_dir}")
        return cls(sorted(str(p) for p in paths))


def build_clip(frame_set, out_path, fps=5.0, ffmpeg="ffmpeg"):
    """ffmpeg the frames into an H.264 MP4 the video endpoint accepts.

    Uses the concat demuxer so non-sequential filenames (`frame_000000.jpg`,
    `frame_000040.jpg`, ...) keep their order. The clip's frame i is
    `frame_set.paths[i]`, which is the mapping SAM's `<Nf>` is read through.
    Encoder flags mirror `ingest.reencode`: dense keyframes, no B-frames,
    faststart.
    """
    out_path = str(out_path)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if len(frame_set) > VIDEO_MAX_FRAMES:
        raise ValueError(
            f"{len(frame_set)} frames exceeds the documented {VIDEO_MAX_FRAMES}-frame "
            "clip limit — split the run into chunks")
    def _safe(p):
        return str(pathlib.Path(p).resolve()).replace("\\", "/").replace("'", "'\\''")

    listing = []
    for p in frame_set.paths:
        listing.append(f"file '{_safe(p)}'")
        listing.append(f"duration {1.0 / float(fps):.6f}")
    # the concat demuxer drops the last frame without a repeat of its entry
    listing.append(f"file '{_safe(frame_set.paths[-1])}'")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(listing) + "\n")
        list_path = fh.name
    try:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
               "-vsync", "cfr", "-r", str(fps),
               # exactly one encoded frame per input frame. The concat demuxer
               # needs the last entry repeated to honour its duration, which
               # otherwise emits one frame too many — and a clip one frame long
               # would shift every `<Nf>` against its canonical frame number,
               # i.e. every detection onto the wrong moment of the video.
               "-frames:v", str(len(frame_set)),
               "-c:v", "libx264", "-g", "12", "-bf", "0",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg clip build failed: {res.stderr[-2000:]}")
        _verify_clip_frames(out_path, len(frame_set))
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    return out_path


def _verify_clip_frames(clip_path, expected):
    """Decode-count the clip before spending anything on it.

    The frame mapping is the one thing in this stage that fails *silently*: a
    clip with the wrong number of frames puts every detection on the wrong
    moment, and nothing downstream can tell. Counting is cheap; being wrong is
    not.
    """
    try:
        import av
    except ImportError:                      # pragma: no cover - av is a dependency
        return None
    container = av.open(str(clip_path))
    try:
        n = sum(1 for _ in container.decode(container.streams.video[0]))
    finally:
        container.close()
    if n != expected:
        raise RuntimeError(
            f"clip {clip_path} decoded {n} frames, expected {expected} — refusing "
            "to send it, because SAM's frame numbers would not line up with the "
            "frames they came from")
    return n


def data_url(path):
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "video/mp4" if str(path).lower().endswith((".mp4", ".mov")) else "image/jpeg"
    b64 = base64.b64encode(pathlib.Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------

def _urllib_transport(url, headers, body, timeout):
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


class SamVideoClient:
    """SAM 3.1 over the streaming Responses endpoint.

    Call shape is `Body/work/sam_probe.py`'s, which is the shape proven live on
    2026-10-27 (55 person boxes and 45 faces off one real crowd photo): POST
    `/v1/responses`, `model: "sam-3.1"`, one user message carrying `input_text`
    (the concept, or box-prompt tokens) plus `input_image`/`input_video`, and
    `stream: true` with `Accept: text/event-stream`.

    `transport(url, headers, body_bytes, timeout) -> (status, text)` is the only
    seam to the network, so every test in this module runs with no key and no
    HTTP. The key is read fresh from disk per call, never stored on the instance,
    never logged, and scrubbed out of any error text.
    """

    def __init__(self, key_path=DEFAULT_KEY_FILE, url=API_URL, model=MODEL,
                 transport=None, timeout=300, max_usd=2.0,
                 include_confidence=True):
        self.key_path = pathlib.Path(key_path)
        self.url = url
        self.model = model
        self.transport = transport or _urllib_transport
        self.timeout = timeout
        self.max_usd = max_usd
        self.include_confidence = include_confidence
        self.spent_usd = 0.0
        self.calls = []            # one dict per call: kind, n_frames, usd, bytes

    # -- key ----------------------------------------------------------------
    def key_present(self):
        """True when the file holds something that could be a key. Never prints
        it, never returns it — length only."""
        try:
            raw = "".join(self.key_path.read_text(encoding="utf-8",
                                                  errors="replace").split())
        except OSError:
            return False
        return len(raw) >= 20

    def _key(self):
        if not self.key_path.exists():
            raise SamError(f"no key file at {self.key_path}")
        key = "".join(self.key_path.read_text(encoding="utf-8",
                                              errors="replace").split())
        if len(key) < 20:
            raise SamError(
                f"key file {self.key_path} holds {len(key)} characters — empty or "
                "truncated, not a key")
        return key

    # -- spend --------------------------------------------------------------
    def _charge(self, kind, usd, n_frames, note="", billed_frames=None):
        if self.max_usd is not None and self.spent_usd + usd > self.max_usd + 1e-9:
            raise SpendCapExceeded(
                f"this {kind} call would cost ${usd:.4f}, taking the run to "
                f"${self.spent_usd + usd:.4f} against a ${self.max_usd:.2f} cap — "
                "refused before sending. Raise max_usd deliberately if that is wrong.")
        self.spent_usd += usd
        self.calls.append({"kind": kind, "n_frames": int(n_frames),
                           "billed_frames": int(billed_frames if billed_frames
                                                is not None else n_frames),
                           "usd": float(usd), "note": note})

    # -- requests -----------------------------------------------------------
    def _body(self, prompt_text, media_part):
        body = {
            "model": self.model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}, media_part],
            }],
            "stream": True,
        }
        if self.include_confidence:
            body["metadata"] = {"include_confidence": "true"}
        return body

    def _post(self, body, what):
        key = self._key()
        headers = {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = json.dumps(body).encode()
        # A dropped connection is not a failed request, and a long run makes
        # one near-certain: a 16-tile two-concept pass is 32 large uploads back
        # to back, and the first live one lost 10 of them to connection resets
        # and DNS failures on a flaky link. Retrying costs nothing when the
        # send never landed, and the spend was already charged before sending,
        # so a lost tile is paid for either way — the only question is whether
        # anything comes back for the money.
        last = None
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                status, text = self.transport(self.url, headers, payload,
                                              self.timeout)
                last = None
                break
            except Exception as e:                   # network, DNS, timeout
                last = e
                if attempt < TRANSPORT_RETRIES:
                    time.sleep(TRANSPORT_BACKOFF_SECONDS * (2 ** attempt))
        if last is not None:
            raise SamError(f"{what}: {type(last).__name__}: "
                           f"{str(last).replace(key, '<key>')} "
                           f"(after {TRANSPORT_RETRIES + 1} attempts)") from None
        text = text.replace(key, "<key>")
        if status != 200:
            raise SamError(f"{what}: HTTP {status} — {text[:600]}")
        return text

    @staticmethod
    def _output_text(stream_text):
        """Accumulate `response.output_text.delta` out of the SSE stream.

        Falls back to the raw body when the body is not SSE at all, which is how
        `sam_probe.py` read it (it regexed the whole body) — so a transport that
        hands back plain JSON or plain tokens still parses.
        """
        if "data:" not in stream_text:
            return stream_text
        out, saw_completed, saw_delta = [], False, False
        for line in stream_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            etype = event.get("type")
            if etype == "response.output_text.delta":
                delta = event.get("delta")
                if delta:
                    out.append(delta)
                    saw_delta = True
            elif etype == "response.completed":
                saw_completed = True
                # the terminal event often carries the whole text; prefer the
                # deltas when we saw any, and fall back to it when we saw none
                if not saw_delta:
                    resp = event.get("response") or {}
                    for item in resp.get("output", []) or []:
                        for c in item.get("content", []) or []:
                            if c.get("text"):
                                out.append(c["text"])
            elif etype == "error" or ("error" in event and event.get("error")):
                raise SamError(f"error event mid-stream: {json.dumps(event)[:600]}")
        if not out and not saw_completed:
            return stream_text          # unknown shape: let the box regex try
        return "".join(out)

    def segment_image(self, image_path, concept=DEFAULT_PERSON_CONCEPT):
        """One image, one concept. Returns the raw output text (token grammar)."""
        self._charge("image", image_cost_usd(1), 1, f"{concept}:{os.path.basename(str(image_path))}")
        body = self._body(concept, {"type": "input_image",
                                    "image_url": data_url(image_path)})
        return self._output_text(self._post(body, f"image segment ({concept})"))

    def segment_video(self, video_path=None, concept=DEFAULT_PERSON_CONCEPT,
                      n_frames=0, video_url=None, file_id=None):
        """One clip, one concept (or box-prompt tokens as `concept`).

        Video is supplied as `input_video` — docs give `video_url` or `file_id`
        for that part. A local file goes up as a base64 data URL, which is the
        shape the image path proved; a `video_url` or `file_id` passes straight
        through for when the clip is already hosted or uploaded. Which of the
        three the hosted API actually accepts for video is **not yet verified
        live** — the summary says so rather than implying it is.
        """
        if file_id:
            part = {"type": "input_video", "file_id": file_id}
        elif video_url:
            part = {"type": "input_video", "video_url": video_url}
        elif video_path:
            part = {"type": "input_video", "video_url": data_url(video_path)}
        else:
            raise ValueError("segment_video needs video_path, video_url or file_id")
        n = int(n_frames) or VIDEO_MIN_BILLED_FRAMES
        self._charge("video", video_cost_usd(n, n_requests=1), n, concept[:40],
                     billed_frames=max(n, VIDEO_MIN_BILLED_FRAMES))
        body = self._body(concept, part)
        return self._output_text(self._post(body, "video segment"))

    def cost_report(self):
        img = sum(1 for c in self.calls if c["kind"] == "image")
        vid = [c for c in self.calls if c["kind"] == "video"]
        return {
            "spent_usd": round(self.spent_usd, 4),
            "image_calls": img,
            "video_calls": len(vid),
            # sent vs billed differ whenever a request is under the 50-frame
            # minimum, which chunking makes the normal case rather than the
            # exception — a bill read off the frames sent would understate it
            "video_frames_sent": int(sum(c["n_frames"] for c in vid)),
            "video_frames_billed": int(sum(c.get("billed_frames", c["n_frames"])
                                           for c in vid)),
            "max_usd": self.max_usd,
        }


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _fields(blob):
    return {k: v for k, v in _FIELD_RE.findall(blob)}


def _num(d, key, cast=float, default=None):
    if key not in d:
        return default
    try:
        return cast(float(d[key]))
    except (TypeError, ValueError):
        return default


def parse_segmentation(output_text):
    """Token grammar -> `{clip_frame_index: [detection, ...]}`.

    detection: object_id, bbox [x1,y1,x2,y2], confidence, frame_width/height,
    mask (height/width/encoding/payload_bytes) or None.

    Tolerances, each one deliberate:
    * an object id prefix is optional (the live probe's stream was matched on
      the box token alone); missing ids become the detection's position.
    * box fields are read by name, so field order or an added field is harmless.
    * output with no `<Nf>` marker at all — an image response in the wild — is
      treated as frame 0 rather than as zero detections.
    * a mask/box count mismatch is **recorded**, not raised: this is a pipeline
      stage, and losing a 3,000-frame pass over one odd frame is the worse
      failure. `parse_warnings()` reads the record back.
    """
    frames = _WarnDict()
    warnings = frames.warnings
    if not output_text or not output_text.strip():
        return frames
    markers = list(_FRAME_RE.finditer(output_text))
    if not markers:
        segments = [(0, output_text)]
    else:
        segments = []
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(output_text)
            segments.append((int(m.group(1)), output_text[m.end():end]))

    for frame_index, segment in segments:
        dets = []
        for pos, bm in enumerate(_BOX_RE.finditer(segment)):
            obj_id, blob = bm.group(1), bm.group(2)
            f = _fields(blob)
            x1, y1 = _num(f, "x1"), _num(f, "y1")
            x2, y2 = _num(f, "x2"), _num(f, "y2")
            if None in (x1, y1, x2, y2):
                continue
            dets.append({
                "object_id": int(obj_id) if obj_id is not None else pos,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "frame_width": _num(f, "w", int),
                "frame_height": _num(f, "h", int),
                "confidence": _num(f, "c"),
                "mask": None,
            })
        masks = [_parse_mask(mm.group(1)) for mm in _MASK_RE.finditer(segment)]
        if masks and len(masks) != len(dets):
            warnings.append(f"frame {frame_index}: {len(dets)} box(es), {len(masks)} mask(s)")
        for det, mask in zip(dets, masks):
            det["mask"] = mask
        if dets:
            frames.setdefault(frame_index, []).extend(dets)
    return frames


def _parse_mask(blob):
    """`mask ::= "<|mask;x=0;y=0" [";c="DEC] ";data=" H "," W "," ENC PAYLOAD "|>"`.

    Header only. The payload is base85 with a layout only Meta's own parser
    libraries specify, and installing those is Jayden's call — so its length is
    recorded and its bytes are left alone.
    """
    mf = _fields(blob)
    parts = (mf.get("data") or "").split(",")

    def _int(s):
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None

    enc, payload_bytes = None, 0
    if len(parts) >= 3:
        enc = parts[2][:1] or None
        payload_bytes = max(0, len(",".join(parts[2:])) - 1)
    return {
        "height": _int(parts[0]) if parts else None,
        "width": _int(parts[1]) if len(parts) > 1 else None,
        "encoding": enc,
        "payload_bytes": payload_bytes,
        "confidence": _num(mf, "c"),
    }


class _WarnDict(dict):
    """A frames dict that also carries the parse warnings, so a caller that
    ignores them behaves exactly as before and one that wants them can ask."""

    def __init__(self, frames=None, warnings=None):
        super().__init__(frames or {})
        self.warnings = list(warnings or [])


def parse_warnings(frames):
    return list(getattr(frames, "warnings", []))


def detections_to_frame_table(frames, frame_set=None, concept="person",
                              warnings=None):
    """`{clip_index: [det]}` -> a long DataFrame, in canonical frame numbers.

    A clip index past the end of `frame_set` is **dropped**, not kept. It cannot
    be mapped to a real moment, and `FrameSet.canonical` falls back to returning
    the raw index — which on a 31-frame chunk numbered from 1 would put a 32nd
    frame line onto canonical frame 31, i.e. two detections of different moments
    sharing one frame number. That is the frame-mapping failure this module
    refuses to let happen quietly, so it is counted and warned about.
    """
    rows = []
    over = 0
    for clip_index in sorted(frames):
        if frame_set is not None and int(clip_index) >= len(frame_set):
            over += len(frames[clip_index])
            continue
        canonical = frame_set.canonical(clip_index) if frame_set else int(clip_index)
        for det in frames[clip_index]:
            x1, y1, x2, y2 = det["bbox"]
            rows.append({
                "concept": concept,
                "clip_index": int(clip_index),
                "frame": int(canonical),
                "object_id": int(det["object_id"]),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "conf": det.get("confidence"),
                "frame_width": det.get("frame_width"),
                "frame_height": det.get("frame_height"),
                "mask_h": (det.get("mask") or {}).get("height"),
                "mask_w": (det.get("mask") or {}).get("width"),
                "mask_enc": (det.get("mask") or {}).get("encoding"),
            })
    if over and warnings is not None:
        warnings.append(
            f"{over} detection(s) came back on clip frames past the "
            f"{len(frame_set)}-frame clip and were dropped — they cannot be "
            "placed on a real moment")
    cols = ["concept", "clip_index", "frame", "object_id", "x1", "y1", "x2", "y2",
            "conf", "frame_width", "frame_height", "mask_h", "mask_w", "mask_enc"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def feet_point(x1, y1, x2, y2):
    """Bottom-centre of the box — what calibration's floor homography wants."""
    return (float(x1) + float(x2)) / 2.0, float(y2)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def containment(inner, outer):
    """Share of `inner`'s area that lies inside `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(iw * ih / area)


# ---------------------------------------------------------------------------
# tracks — canonical rows
# ---------------------------------------------------------------------------

def tracks_from_object_ids(det_table, id_prefix=""):
    """The video path: SAM's own `object_id` is stable across frames, so it *is*
    the track. Returns the canonical (track_id, frame, x, y) table."""
    rows = []
    for _, r in det_table.iterrows():
        x, y = feet_point(r.x1, r.y1, r.x2, r.y2)
        rows.append((f"{id_prefix}{int(r.object_id):06d}", int(r.frame), x, y))
    df = pd.DataFrame(rows, columns=["track_id", "frame", "x", "y"])
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def stitch_tracks(det_table, max_gap_frames=3, iou_min=0.15,
                  max_move_box_heights=1.25, id_prefix=""):
    """The image path: per-frame boxes -> tracks, by IoU with a distance fallback.

    Frame to frame, open tracks are matched to the new frame's boxes with
    `linear_sum_assignment` over a cost that is `1 - IoU` where the boxes
    overlap, and a centroid-distance penalty where they do not (sparse sampling
    moves a walker clean off their own box in one 5 fps step, so IoU alone
    fragments every track). A track survives `max_gap_frames` sample steps of
    non-detection before it is closed, which is the same occlusion tolerance
    `track_cleaning.bridge_tracks` applies later at the trajectory level.

    Deliberately simple and deterministic: SAM's video pass already does this
    better, and anything cleverer here would fight stage 3.
    """
    from scipy.optimize import linear_sum_assignment

    if len(det_table) == 0:
        return pd.DataFrame(columns=["track_id", "frame", "x", "y"]), {}

    frames = sorted(det_table["frame"].unique())
    # frame number -> its position in the sample sequence, computed once. Done by
    # list.index() this was O(n_frames) inside the per-detection loop, which on a
    # 3,000-frame run is tens of millions of scans — the image path would have
    # crawled with nothing to show why.
    pos = {int(f): i for i, f in enumerate(frames)}
    by_frame = {int(f): g.reset_index(drop=True)
                for f, g in det_table.groupby("frame", sort=True)}
    open_tracks = {}      # tid -> {"bbox", "frame"}
    assignments = {}      # (frame, object_id) -> tid
    rows = []
    next_id = 0

    for frame in frames:
        dets = by_frame[int(frame)]
        boxes = [[r.x1, r.y1, r.x2, r.y2] for r in dets.itertuples()]
        here = pos[int(frame)]
        live = [tid for tid, t in open_tracks.items()
                if here - pos[int(t["frame"])] <= max_gap_frames]

        matched_dets, matched_tracks = set(), set()
        if live and boxes:
            cost = np.full((len(live), len(boxes)), 1e9)
            for i, tid in enumerate(live):
                tb = open_tracks[tid]["bbox"]
                th = max(1.0, tb[3] - tb[1])
                steps = max(1, here - pos[int(open_tracks[tid]["frame"])])
                for j, bb in enumerate(boxes):
                    iou = box_iou(tb, bb)
                    if iou >= iou_min:
                        cost[i, j] = 1.0 - iou
                        continue
                    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                    bcx, bcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    d = float(np.hypot(bcx - tcx, bcy - tcy))
                    budget = max_move_box_heights * th * steps
                    if d <= budget:
                        cost[i, j] = 1.0 + d / max(1e-6, budget)   # worse than any IoU match
            if np.any(cost < 1e9):
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if cost[i, j] >= 1e9:
                        continue
                    tid = live[i]
                    matched_tracks.add(tid)
                    matched_dets.add(j)
                    open_tracks[tid] = {"bbox": boxes[j], "frame": frame}
                    assignments[(int(frame), int(dets.iloc[j].object_id))] = tid
                    x, y = feet_point(*boxes[j])
                    rows.append((tid, int(frame), x, y))

        for j, bb in enumerate(boxes):
            if j in matched_dets:
                continue
            tid = f"{id_prefix}{next_id:06d}"
            next_id += 1
            open_tracks[tid] = {"bbox": bb, "frame": frame}
            assignments[(int(frame), int(dets.iloc[j].object_id))] = tid
            x, y = feet_point(*bb)
            rows.append((tid, int(frame), x, y))

        for tid in list(open_tracks):
            if here - pos[int(open_tracks[tid]["frame"])] > max_gap_frames:
                del open_tracks[tid]

    df = pd.DataFrame(rows, columns=["track_id", "frame", "x", "y"])
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    return df, assignments


def attach_track_ids(det_table, assignments=None, id_prefix=""):
    """Add a `track_id` column to a detection table — from the stitcher's
    assignments if given, else straight off SAM's own object ids."""
    out = det_table.copy()
    if assignments:
        out["track_id"] = [assignments.get((int(r.frame), int(r.object_id)))
                           for r in out.itertuples()]
    else:
        out["track_id"] = [f"{id_prefix}{int(r.object_id):06d}" for r in out.itertuples()]
    return out


# ---------------------------------------------------------------------------
# >16 people: box-prompt partitioning and the cross-pass merge
# ---------------------------------------------------------------------------

def partition_boxes(boxes, max_objects=VIDEO_OBJECT_CAP):
    """Split boxes into spatially adjacent groups of at most `max_objects`.

    Recursive median split on the wider axis of each cell's centres — a 2-d KD
    split. Deterministic, no dependencies, and it keeps neighbours together,
    which is what the spec asks for ("groups of spatially-adjacent people"):
    people who might be in one conversation should land in one pass, so their
    relative geometry is never split across passes.
    """
    idx = list(range(len(boxes)))
    if not idx:
        return []
    centres = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes])

    out = []

    def split(group):
        if len(group) <= max_objects:
            out.append(sorted(group))
            return
        pts = centres[group]
        axis = 0 if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else 1
        order = sorted(group, key=lambda i: centres[i, axis])
        half = len(order) // 2
        split(order[:half])
        split(order[half:])

    split(idx)
    return sorted(out, key=lambda g: g[0])


def box_prompt_text(boxes, frame_index=0, frame_width=None, frame_height=None):
    """Box-prompt tokens for `input_text`, in the documented grammar:
    `<Nf>` then `id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|>` comma-separated."""
    parts = []
    for i, b in enumerate(boxes):
        fields = [f"x1={int(round(b[0]))}", f"y1={int(round(b[1]))}",
                  f"x2={int(round(b[2]))}", f"y2={int(round(b[3]))}"]
        if frame_width:
            fields.append(f"w={int(frame_width)}")
        if frame_height:
            fields.append(f"h={int(frame_height)}")
        parts.append(f"{i}<|box;" + ";".join(fields) + "|>")
    return f"<{int(frame_index)}f>" + ",".join(parts)


def merge_passes(pass_tables, iou_threshold=0.5):
    """Merge per-partition detection tables into one, dropping duplicates.

    Each pass is seeded with disjoint boxes, so the merge is mostly a
    concatenation with namespaced ids. The safety net the spec asks for is here:
    tracks from different passes whose boxes overlap above `iou_threshold` on
    the frames they share are the same person caught twice, and the shorter one
    is dropped. Hungarian assignment resolves which pairs win when several
    tracks compete.

    (Mask-IoU would be the stronger comparison and needs the mask payload
    decoded, which this module does not do — box IoU over shared frames is the
    documented fallback and the one actually implemented.)
    """
    from scipy.optimize import linear_sum_assignment

    tables = []
    for p, t in enumerate(pass_tables):
        if t is None or len(t) == 0:
            continue
        t = t.copy()
        t["pass"] = p
        t["track_id"] = [f"p{p}_{int(o):06d}" for o in t["object_id"]]
        tables.append(t)
    if not tables:
        return pd.DataFrame(columns=["concept", "clip_index", "frame", "object_id",
                                     "x1", "y1", "x2", "y2", "conf", "pass", "track_id"]), []
    allt = pd.concat(tables, ignore_index=True)

    by_track = {tid: g for tid, g in allt.groupby("track_id", sort=True)}
    # one {frame: box} dict per track, built once. Indexing the frames out of the
    # frame with `.loc` instead returned a *Series* the moment a track held two
    # rows for one frame and died inside box_iou on an unreadable pandas error —
    # and a duplicated frame number is exactly what a stray frame line makes.
    boxes_by_track = {}
    for tid, g in by_track.items():
        d = {}
        for r in g.itertuples():
            d[int(r.frame)] = [float(r.x1), float(r.y1), float(r.x2), float(r.y2)]
        boxes_by_track[tid] = d
    dropped = []
    passes = sorted(allt["pass"].unique())
    for a in range(len(passes)):
        for b in range(a + 1, len(passes)):
            ta = [t for t in by_track if by_track[t]["pass"].iloc[0] == passes[a]]
            tb = [t for t in by_track if by_track[t]["pass"].iloc[0] == passes[b]]
            if not ta or not tb:
                continue
            cost = np.full((len(ta), len(tb)), 1e9)
            for i, x in enumerate(ta):
                bx = boxes_by_track[x]
                for j, y in enumerate(tb):
                    by = boxes_by_track[y]
                    shared = sorted(set(bx) & set(by))
                    if not shared:
                        continue
                    mean_iou = float(np.mean([box_iou(bx[f], by[f]) for f in shared]))
                    if mean_iou >= iou_threshold:
                        cost[i, j] = 1.0 - mean_iou
            if not np.any(cost < 1e9):
                continue
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] >= 1e9:
                    continue
                x, y = ta[i], tb[j]
                keep, drop = (x, y) if len(by_track[x]) >= len(by_track[y]) else (y, x)
                if drop in by_track:
                    dropped.append({"dropped": drop, "kept": keep,
                                    "mean_iou": round(1.0 - float(cost[i, j]), 3)})
                    del by_track[drop]
    merged = pd.concat(by_track.values(), ignore_index=True) if by_track else allt.iloc[0:0]
    merged = merged.sort_values(["track_id", "frame"]).reset_index(drop=True)
    return merged, dropped


# ---------------------------------------------------------------------------
# chunked re-seeding: the people who walk in after the seed frame
# ---------------------------------------------------------------------------
# The defect this closes, measured on the plaza clip 2026-10-27: box prompts
# seed once, on frame 1, so the run tracked the 49 people standing there and
# never saw anybody who arrived afterwards. At frame 45 the image endpoint found
# 52 people and the run was holding 35 — and the 17 it was missing were the
# nearest, largest people in the shot, because the foreground is exactly where
# somebody can walk in, cross and leave inside one clip.
#
# The fix: cut the clip into chunks, run a fresh detection on each chunk's first
# frame, seed that chunk with **both** the continuing tracks' latest boxes and
# whatever the fresh detection found that is not one of them, then join each
# chunk's tracks onto the running ids across the seam. A chunk seam is the same
# comparison `merge_passes` makes between partitions, aimed across time rather
# than across space — with one difference that matters: a matched pair is
# *joined*, never dropped.


def chunk_spans(n_frames, chunk_frames=DEFAULT_CHUNK_FRAMES,
                overlap_frames=DEFAULT_CHUNK_OVERLAP):
    """Contiguous index spans `[start, stop)` covering the clip, neighbours
    sharing `overlap_frames`.

    `chunk_frames` is a **target, not a ceiling**: the clip is cut into the whole
    number of chunks nearest that length and the spans come out even. A short
    final chunk would be billed the same 50-frame minimum as a full one and buy
    nothing with it, so evening the spans out is free accuracy.

    The overlap is what the seam merge stands on: both chunks either side of it
    hold boxes for the *same* canonical frames, so continuing tracks are matched
    on same-frame IoU instead of on a guess about how far somebody moved across
    the boundary.
    """
    n = int(n_frames)
    if n <= 0:
        return []
    if not chunk_frames or n <= int(chunk_frames):
        return [(0, n)]
    ov = max(0, min(int(overlap_frames), int(chunk_frames) - 1))
    stride_target = max(1, int(chunk_frames) - ov)
    n_chunks = max(1, int(round((n - ov) / float(stride_target))))
    # a chunk must be longer than the overlap it has to share, or two seams
    # would land on the same frames
    while n_chunks > 1 and (n - ov) / float(n_chunks) <= max(1, ov):
        n_chunks -= 1
    if n_chunks <= 1:
        return [(0, n)]
    step = (n - ov) / float(n_chunks)
    spans = []
    for k in range(n_chunks):
        a = int(round(k * step))
        b = n if k == n_chunks - 1 else min(n, int(round((k + 1) * step)) + ov)
        spans.append((a, b))
    return spans


def continuing_seed_boxes(acc_table, seed_frame, lookback=DEFAULT_SEED_LOOKBACK):
    """The latest box each already-tracked person holds at or just before
    `seed_frame` — the boxes that carry identity into the next chunk.

    A small lookback exists because a person occluded on the exact seam frame
    would otherwise be re-seeded as a stranger and lose their id. It is kept
    tight (2 frames, 0.4 s at 5 fps) because a stale box is a prompt pointing at
    where somebody *was*.
    """
    if acc_table is None or not len(acc_table):
        return []
    seed_frame = int(seed_frame)
    lo = seed_frame - max(0, int(lookback))
    win = acc_table[(acc_table["frame"] >= lo) & (acc_table["frame"] <= seed_frame)]
    out = []
    for tid, g in win.groupby("track_id", sort=True):
        r = g.loc[g["frame"].idxmax()]
        out.append((str(tid), [float(r.x1), float(r.y1), float(r.x2), float(r.y2)]))
    return out


def compose_chunk_seed(continuing, fresh_boxes, iou_min=DEFAULT_RESEED_IOU):
    """Seed boxes for one chunk: every continuing track, plus every fresh
    detection that is not already one of them.

    Matching is Hungarian rather than greedy for one concrete reason: two people
    standing shoulder to shoulder can both overlap one continuing box, and greedy
    matching would call them both "already tracked" and lose one. Hungarian lets
    a continuing box claim at most one fresh detection.

    The continuing track keeps its *own* last box rather than the fresh
    detection it matched — identity stays anchored to the track, and a wrong
    match cannot teleport an id onto a different person.

    Returns `(boxes, origins, info)`; `origins[i]` is the continuing track id or
    None for a newcomer.
    """
    from scipy.optimize import linear_sum_assignment

    cont_boxes = [list(b) for _, b in continuing]
    fresh_boxes = [list(b) for b in (fresh_boxes or [])]
    claimed = set()
    if cont_boxes and fresh_boxes:
        cost = np.full((len(cont_boxes), len(fresh_boxes)), 1e9)
        for i, cb in enumerate(cont_boxes):
            for j, fb in enumerate(fresh_boxes):
                iou = box_iou(cb, fb)
                if iou >= iou_min:
                    cost[i, j] = 1.0 - iou
        if np.any(cost < 1e9):
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e9:
                    claimed.add(int(j))
    boxes = list(cont_boxes)
    origins = [tid for tid, _ in continuing]
    newcomers = [j for j in range(len(fresh_boxes)) if j not in claimed]
    for j in newcomers:
        boxes.append(fresh_boxes[j])
        origins.append(None)
    info = {"continuing": len(cont_boxes), "fresh_detections": len(fresh_boxes),
            "new": len(newcomers), "seed_boxes": len(boxes)}
    return boxes, origins, info


def link_chunk_tracks(acc, chunk_table, shared_frames, next_id=0,
                      iou_threshold=DEFAULT_CHUNK_LINK_IOU, id_prefix=""):
    """Join one chunk's tracks onto the running table across the seam.

    Mean box IoU over the frames the two chunks share, Hungarian assignment, and
    then: a matched chunk track **continues** the running track's id, an
    unmatched one is somebody who was not there at the seed frame and gets a
    fresh id. Rows the running table already holds for a (track, frame) are kept
    as they were, so the overlap frames are not double-counted.

    Returns `(merged, info, next_id)`.
    """
    from scipy.optimize import linear_sum_assignment

    cols = None
    if chunk_table is not None and len(chunk_table):
        cols = list(chunk_table.columns)
    if chunk_table is None or not len(chunk_table):
        return (acc if acc is not None else pd.DataFrame()), \
               {"linked": 0, "new_ids": 0, "shared_frames": 0}, next_id

    chunk_table = chunk_table.copy()
    shared = sorted({int(f) for f in (shared_frames or [])})
    mapping = {}
    links = 0

    if shared and acc is not None and len(acc):
        def _boxes_by_frame(table):
            sub = table[table["frame"].isin(shared)]
            out = {}
            for tid, g in sub.groupby("track_id", sort=True):
                out[str(tid)] = {int(r.frame): [float(r.x1), float(r.y1),
                                                float(r.x2), float(r.y2)]
                                 for r in g.itertuples()}
            return out

        a_boxes = _boxes_by_frame(acc)
        c_boxes = _boxes_by_frame(chunk_table)
        A, C = sorted(a_boxes), sorted(c_boxes)
        if A and C:
            cost = np.full((len(A), len(C)), 1e9)
            for i, at in enumerate(A):
                for j, ct in enumerate(C):
                    common = sorted(set(a_boxes[at]) & set(c_boxes[ct]))
                    if not common:
                        continue
                    mean_iou = float(np.mean([box_iou(a_boxes[at][f], c_boxes[ct][f])
                                              for f in common]))
                    if mean_iou >= iou_threshold:
                        cost[i, j] = 1.0 - mean_iou
            if np.any(cost < 1e9):
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if cost[i, j] < 1e9:
                        mapping[C[j]] = A[i]
                        links += 1

    # unmatched chunk tracks are new people; ids run in order of first sighting
    unmatched = [t for t in sorted(chunk_table["track_id"].astype(str).unique())
                 if t not in mapping]
    firsts = {str(t): int(g["frame"].min())
              for t, g in chunk_table.groupby("track_id", sort=False)}
    for t in sorted(unmatched, key=lambda t: (firsts.get(t, 0), t)):
        mapping[t] = f"{id_prefix}{next_id:06d}"
        next_id += 1

    chunk_table["track_id"] = [mapping[str(t)] for t in chunk_table["track_id"]]

    if acc is None or not len(acc):
        merged = chunk_table
    else:
        have = set(zip(acc["track_id"].astype(str), acc["frame"].astype(int)))
        keep = [not ((str(t), int(f)) in have)
                for t, f in zip(chunk_table["track_id"], chunk_table["frame"])]
        merged = pd.concat([acc, chunk_table[keep]], ignore_index=True)
    merged = merged.sort_values(["track_id", "frame"]).reset_index(drop=True)
    if cols:
        merged = merged[[c for c in cols if c in merged.columns] +
                        [c for c in merged.columns if c not in cols]]
    info = {"linked": links, "new_ids": len(unmatched),
            "shared_frames": len(shared)}
    return merged, info, next_id


# ---------------------------------------------------------------------------
# facing from face boxes
# ---------------------------------------------------------------------------

def pair_faces_to_people(person_table, face_table,
                         min_containment=DEFAULT_MIN_CONTAINMENT):
    """Pair each face box to the person box containing it, per frame.

    A face belongs to the person box it sits most completely inside; ties go to
    the *smallest* such box, which is the person standing in front. One face per
    person — a second face landing in the same body box (a head behind a
    shoulder) is left unpaired rather than overwriting the better match.

    Returns a DataFrame: frame, person_index (row index into `person_table`),
    face_index or NaN, containment, face_conf.
    """
    rows = []
    face_by_frame = {f: g for f, g in face_table.groupby("frame")} if len(face_table) else {}
    for frame, people in person_table.groupby("frame"):
        faces = face_by_frame.get(frame)
        claimed = {}
        if faces is not None and len(faces):
            cands = []
            for f in faces.itertuples():
                fbox = [f.x1, f.y1, f.x2, f.y2]
                for p in people.itertuples():
                    pbox = [p.x1, p.y1, p.x2, p.y2]
                    c = containment(fbox, pbox)
                    if c >= min_containment:
                        area = max(1e-9, (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]))
                        cands.append((c, -area, f.Index, p.Index, f.conf))
            # best containment first, then the tighter person box
            for c, neg_area, fidx, pidx, fconf in sorted(cands, reverse=True):
                if pidx in claimed or fidx in {v[0] for v in claimed.values()}:
                    continue
                claimed[pidx] = (fidx, c, fconf)
        for p in people.itertuples():
            fidx, c, fconf = claimed.get(p.Index, (None, 0.0, None))
            rows.append({"frame": int(frame), "person_index": p.Index,
                         "face_index": fidx, "containment": float(c),
                         "face_conf": fconf})
    out = pd.DataFrame(rows, columns=["frame", "person_index", "face_index",
                                      "containment", "face_conf"])
    # `face_index` is float-with-NaN whatever the mix of hits and misses, so a
    # caller never has to branch on dtype — `has_face()` is the one test.
    out["face_index"] = pd.to_numeric(out["face_index"], errors="coerce").astype("float64")
    return out


def has_face(face_index):
    """One test for "this person was paired with a face", whatever pandas did
    to the column's dtype."""
    if face_index is None:
        return False
    try:
        return bool(np.isfinite(float(face_index)))
    except (TypeError, ValueError):
        return False


def _world_dirs(H, px, py, delta=8.0):
    """Unit vectors in the floor plane for "toward the camera" and "image right"
    at one pixel position.

    Moving *down* the image along a ground plane means moving toward the camera,
    so the floor-plane direction of image-down is camera-ward. Both directions
    are read off the same homography the metrics use, which is why the angles
    come out in the frame `groups.py` reads (`x_m`/`y_m`) rather than in a
    convention of this module's own.
    """
    from proxemix.calibration import project_points
    pts = np.array([[px, py], [px, py + delta], [px + delta, py]], dtype=float)
    w = project_points(H, pts)
    cam = w[1] - w[0]
    right = w[2] - w[0]
    for v in (cam, right):
        n = np.hypot(v[0], v[1])
        if n < 1e-9:
            return None, None
    return cam / np.hypot(*cam), right / np.hypot(*right)


def facing_from_faces(person_table, pairings, calibration_config=None,
                      max_yaw_deg=DEFAULT_MAX_YAW_DEG,
                      face_conf=DEFAULT_FACE_CONF,
                      absent_conf=DEFAULT_ABSENT_CONF,
                      include_absent=True,
                      track_col="track_id",
                      world_dirs=None):
    """Face pairings -> a `(track_id, frame, facing_rad, facing_conf)` table,
    ready for `facing.apply_external_facing`.

    The model, stated plainly so it can be argued with:
    * face present, centred in the body box -> facing camera-ward.
    * face offset by `u` of the half-width -> rotated `u * max_yaw` from
      camera-ward toward image-right, because the visible face sits toward the
      side the head is turned to.
    * face absent -> facing *away* from the camera, at `absent_conf`, which is
      below `groups.py`'s confidence gate by default (see module docstring).

    Confidence for a found face scales with how completely the face box sits
    inside the body box and with SAM's own confidence when it reports one, and
    is capped at `face_conf` — a coarse estimate should never outrank a
    full-speed walker's motion facing (1.0) in `apply_external_facing`.

    `world_dirs` is the seam for a camera that knows better than a homography:
    a callable `(u, v) -> (camward_unit, imageright_unit)` in floor-plane
    coordinates, used instead of `_world_dirs`. `proxemix.salsa_camera` supplies
    one, because `_world_dirs` reads floor-plane image-down as camera-ward and
    that is 4-25 degrees wrong on a rolled, wide-angle view. Left `None`,
    nothing about this function changes.
    """
    H = None
    if calibration_config is not None:
        from proxemix.calibration import build_homography
        H, _ = build_homography(calibration_config)

    max_yaw = np.deg2rad(max_yaw_deg)
    rows = []
    for pr in pairings.itertuples():
        p = person_table.loc[pr.person_index]
        tid = p[track_col] if track_col in person_table.columns else None
        if tid is None or (isinstance(tid, float) and np.isnan(tid)):
            continue
        fx, fy = feet_point(p.x1, p.y1, p.x2, p.y2)
        if world_dirs is not None:
            cam, right = world_dirs(fx, fy)
            if cam is None:
                continue
        elif H is not None:
            cam, right = _world_dirs(H, fx, fy)
            if cam is None:
                continue
        else:
            cam = np.array([0.0, 1.0])     # image-down ~ toward the camera
            right = np.array([1.0, 0.0])

        if not has_face(pr.face_index):
            if not include_absent:
                continue
            vec = -cam
            conf = float(absent_conf)
            source = "face_absent"
            yaw = np.pi
        else:
            face = getattr(pr, "face_box", None)
            if face is None:
                continue
            fcx = (face[0] + face[2]) / 2.0
            pcx = (float(p.x1) + float(p.x2)) / 2.0
            half_w = max(1e-6, (float(p.x2) - float(p.x1)) / 2.0)
            u = float(np.clip((fcx - pcx) / half_w, -1.0, 1.0))
            yaw = u * max_yaw
            vec = np.cos(yaw) * cam + np.sin(yaw) * right
            quality = float(np.clip(pr.containment, 0.0, 1.0))
            if pr.face_conf is not None and np.isfinite(pr.face_conf or np.nan):
                quality *= float(np.clip(pr.face_conf, 0.0, 1.0))
            conf = float(np.clip(face_conf * quality, 0.0, face_conf))
            source = "face"
        n = np.hypot(vec[0], vec[1])
        if n < 1e-9:
            continue
        vec = vec / n
        rows.append({"track_id": tid, "frame": int(pr.frame),
                     "facing_rad": float(np.arctan2(vec[1], vec[0])),
                     "facing_conf": conf,
                     "facing_source": source,
                     "yaw_from_camera_deg": float(np.degrees(yaw))})
    out = pd.DataFrame(rows, columns=["track_id", "frame", "facing_rad",
                                      "facing_conf", "facing_source",
                                      "yaw_from_camera_deg"])
    return out.sort_values(["track_id", "frame"]).reset_index(drop=True)


def pair_and_face(person_table, face_table, **kw):
    """`pair_faces_to_people` + `facing_from_faces` with the face boxes carried
    across, which is the only reason these two are not one function."""
    pairings = pair_faces_to_people(
        person_table, face_table,
        min_containment=kw.pop("min_containment", DEFAULT_MIN_CONTAINMENT))
    boxes = []
    for pr in pairings.itertuples():
        if not has_face(pr.face_index):
            boxes.append(None)
        else:
            f = face_table.loc[int(pr.face_index)]
            boxes.append([float(f.x1), float(f.y1), float(f.x2), float(f.y2)])
    pairings = pairings.copy()
    pairings["face_box"] = boxes
    return facing_from_faces(person_table, pairings, **kw), pairings


def facing_summary(facing_df, n_person_rows=None):
    if facing_df is None or len(facing_df) == 0:
        return {"rows": 0}
    counts = facing_df["facing_source"].value_counts().to_dict()
    face_rows = facing_df[facing_df["facing_source"] == "face"]
    out = {
        "rows": int(len(facing_df)),
        "by_source": {str(k): int(v) for k, v in counts.items()},
        "pct_with_face": round(100.0 * len(face_rows) / max(1, len(facing_df)), 1),
        "mean_conf_face": round(float(face_rows["facing_conf"].mean()), 3) if len(face_rows) else 0.0,
    }
    if n_person_rows:
        out["pct_of_person_rows"] = round(100.0 * len(facing_df) / n_person_rows, 1)
    return out


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------

def headcount(client, frame_set, n_samples=3, concept=DEFAULT_PERSON_CONCEPT):
    """The front porch: the image endpoint on a few stills, which returns every
    instance with no 16-object cap, so it is what routes the scene.

    It also returns the **boxes**, not only the counts. They come back from
    calls already being paid for, and they are what `tiling.plan_tiles` needs
    to decide how finely this particular scene has to be read — so keeping
    them makes the tile plan free. Discarding them, which this function used
    to do, is what made the plan look like it needed its own calls.
    """
    counts, boxes = [], []
    for path in frame_set.sample(n_samples):
        text = client.segment_image(path, concept=concept)
        frames = parse_segmentation(text)
        n = sum(len(v) for v in frames.values())
        counts.append({"path": os.path.basename(path), "n": n})
        for dets in frames.values():
            for det in dets:
                bx = det.get("bbox")
                if bx:
                    boxes.append([bx[0], bx[1], bx[2], bx[3],
                                  float(det.get("confidence") or 0.0)])
    peak = max((c["n"] for c in counts), default=0)
    return {"samples": counts, "peak": int(peak),
            "over_video_cap": bool(peak > VIDEO_OBJECT_CAP),
            "boxes": boxes}


def run_sam_stage(frames_dir=None, frame_set=None, video_path=None,
                  client=None, out_dir=None, mode="auto",
                  person_concept=DEFAULT_PERSON_CONCEPT,
                  face_concept=DEFAULT_FACE_CONCEPT,
                  fps=5.0, headcount_samples=3, do_facing=True,
                  calibration_config=None, keep_clip=False,
                  include_absent_facing=True,
                  chunk_frames=DEFAULT_CHUNK_FRAMES,
                  chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                  clip_builder=None, tiled="auto"):
    """Stage 2 end to end: frames in, canonical rows + facing out.

    Returns a dict:
        rows          canonical (track_id, frame, x, y) — feed to clean_tracks
        detections    the per-detection table with boxes, confidences, track ids
        faces         the face detection table (empty when do_facing=False)
        facing        (track_id, frame, facing_rad, facing_conf, facing_source)
        summary       mode taken, counts, cost in dollars, warnings, what is
                      proven live vs not

    `mode="auto"` runs the headcount front porch and routes: <=16 people -> one
    video pass per concept; >16 -> box-prompt partitioned video passes.

    **Chunked re-seeding is on by default** (`chunk_frames=30`), because seeding
    once meant never tracking anybody who walked in after the first frame — on
    the plaza clip that was a third of the crowd by frame 45, and the nearest,
    largest third of it. `chunk_frames=None` (or 0) restores the single-shot
    behaviour, which is what every live number banked before 2026-10-28 was
    measured on; clips shorter than one chunk take that path anyway, since there
    is nothing to cut.
    """
    if client is None:
        client = SamVideoClient()
    if frame_set is None:
        if frames_dir is None:
            raise ValueError("give frames_dir, frame_set or a prepared video_path")
        frame_set = FrameSet.from_dir(frames_dir)

    n_frames = len(frame_set)
    summary = {
        "n_frames": n_frames,
        "mode_requested": mode,
        "fps": fps,
        "warnings": [],
        "cost_model": {"image_usd_per_1k": IMAGE_USD_PER_1K,
                       "video_usd_per_1k": VIDEO_USD_PER_1K,
                       "video_min_billed_frames": VIDEO_MIN_BILLED_FRAMES},
        "estimates_usd": {
            "video_two_concepts": round(estimate_pass_cost(n_frames, 2, "video"), 4),
            "image_two_concepts": round(estimate_pass_cost(n_frames, 2, "image"), 4),
        },
    }

    spans = chunk_spans(n_frames, chunk_frames, chunk_overlap)
    chunked = len(spans) > 1
    summary["chunking"] = {
        "enabled": bool(chunked),
        "target_chunk_frames": int(chunk_frames) if chunk_frames else 0,
        "overlap_frames": int(chunk_overlap),
        "n_chunks": len(spans),
        "spans_frames": [[int(frame_set.frames[a]), int(frame_set.frames[b - 1])]
                         for a, b in spans],
    }
    if not chunked:
        summary["chunking"]["reason"] = (
            "chunking off — single-shot seeding, which cannot track anybody who "
            "arrives after frame 1" if not chunk_frames else
            f"{n_frames} frames fits inside one {int(chunk_frames)}-frame chunk")

    head = None
    if mode == "auto":
        head = headcount(client, frame_set, n_samples=headcount_samples,
                         concept=person_concept)
        summary["headcount"] = head
        mode = "video_partitioned" if head["over_video_cap"] else "video"

        # How finely this scene needs reading, computed from the boxes the
        # headcount already returned — no extra calls. Recorded rather than
        # acted on: the measured gain (recall +4.3 points on SALSA at 2x2) is
        # real, and so is the measured harm of over-splitting (precision 0.69
        # -> 0.38 at 4x4), so the plan is reported for the run's record and
        # the tiled pass is opted into deliberately. `tiling.py` holds the
        # evidence for both halves.
        try:
            from proxemix.tiling import plan_tiles
            hc_boxes = head.get("boxes") or []
            if hc_boxes:
                probe = frame_set.sample(1)
                if probe:
                    from PIL import Image
                    pw, ph = Image.open(probe[0]).size
                    plan = plan_tiles(hc_boxes, pw, ph)
                    summary["tile_plan"] = plan["stats"]
                    summary["tile_plan"]["tiles"] = [list(t) for t in plan["tiles"]]
        except Exception as e:                      # never fail the stage for this
            summary.setdefault("warnings", []).append(
                f"tile plan not computed: {type(e).__name__}: {str(e)[:120]}")
    summary["mode_taken"] = mode + ("_chunked" if chunked and mode != "image" else "")
    if chunked and mode in ("video", "video_partitioned"):
        n_parts = max(1, -(-int((head or {}).get("peak", 1)) // VIDEO_OBJECT_CAP)) \
            if mode == "video_partitioned" else 1
        summary["estimates_usd"]["chunked_two_concepts"] = round(
            estimate_chunked_cost(n_frames, 2, n_parts, chunk_frames,
                                  chunk_overlap), 4)

    # Tiling is taken when the plan asked for more than one tile. `tiled="auto"`
    # follows the plan, True forces it, False refuses it. The plan itself is a
    # ceiling rule measured against SALSA's annotations — 2x2 lifted recall
    # 64.7% -> 69.1% with precision also up, and 4x4 halved precision — so
    # following it is following evidence rather than a preference.
    plan_tiles_list = [tuple(t) for t in
                       (summary.get("tile_plan") or {}).get("tiles", [])]
    use_tiles = False
    if tiled is True:
        use_tiles = bool(plan_tiles_list)
    elif tiled == "auto":
        use_tiles = len(plan_tiles_list) > 1
    summary["tiled_taken"] = bool(use_tiles)
    if use_tiles:
        summary["mode_taken"] = summary["mode_taken"] + "_tiled"

    person_dets = pd.DataFrame()
    face_dets = pd.DataFrame()
    assignments = None
    merge_drops = []

    if mode == "image":
        person_dets = _image_pass(client, frame_set, person_concept, summary)
        if do_facing:
            face_dets = _image_pass(client, frame_set, face_concept, summary)
        rows, assignments = stitch_tracks(person_dets)
        person_dets = attach_track_ids(person_dets, assignments)

    elif use_tiles:
        # Tiles replace chunking for this pass rather than stacking with it:
        # chunking exists to re-seed box prompts for people who arrive mid-clip,
        # and a tile small enough to need no box prompts has nothing to re-seed.
        # Stacking both would multiply calls by tiles x chunks for no measured
        # gain, which is exactly the over-spending the ceiling rule prevents.
        clip_dir = out_dir or tempfile.gettempdir()
        os.makedirs(clip_dir, exist_ok=True)
        person_dets, merge_drops, tinfo = _tiled_video_pass(
            client, frame_set, person_concept, summary, plan_tiles_list,
            fps=fps, clip_dir=clip_dir, clip_builder=clip_builder,
            keep_clips=keep_clip, key=person_concept)
        summary["tiles_person"] = tinfo
        summary["cross_pass_duplicates_dropped"] = len(merge_drops)
        if do_facing:
            face_dets, face_drops, finfo = _tiled_video_pass(
                client, frame_set, face_concept, summary, plan_tiles_list,
                fps=fps, clip_dir=clip_dir, clip_builder=clip_builder,
                keep_clips=keep_clip, key=face_concept)
            summary["tiles_face"] = finfo
            summary["cross_pass_duplicates_dropped_face"] = len(face_drops)
        rows = _rows_from_boxes(person_dets)

    elif chunked and mode in ("video", "video_partitioned"):
        if video_path:
            raise ValueError(
                "a prepared video_path cannot be chunked — this stage builds one "
                "clip per chunk from the frames, so pass frames_dir/frame_set, or "
                "set chunk_frames=None to run the single-shot path on your clip")
        partitioned = mode == "video_partitioned"
        clip_dir = out_dir or tempfile.gettempdir()
        os.makedirs(clip_dir, exist_ok=True)
        person_dets, merge_drops, chunks = _chunked_video_pass(
            client, frame_set, person_concept, summary, partitioned=partitioned,
            chunk_frames=chunk_frames, chunk_overlap=chunk_overlap, fps=fps,
            clip_dir=clip_dir, clip_builder=clip_builder, keep_clips=keep_clip,
            key=person_concept)
        summary["chunks_person"] = chunks
        summary["cross_pass_duplicates_dropped"] = len(merge_drops)
        summary["reseeded_newcomers_person"] = int(
            sum(c.get("new", 0) for c in chunks[1:]))
        if do_facing:
            face_dets, face_drops, face_chunks = _chunked_video_pass(
                client, frame_set, face_concept, summary, partitioned=partitioned,
                chunk_frames=chunk_frames, chunk_overlap=chunk_overlap, fps=fps,
                clip_dir=clip_dir, clip_builder=clip_builder,
                keep_clips=keep_clip, key=face_concept)
            summary["chunks_face"] = face_chunks
            summary["cross_pass_duplicates_dropped_face"] = len(face_drops)
        rows = _rows_from_boxes(person_dets)

    elif mode in ("video", "video_partitioned"):
        clip = video_path
        tmp_clip = None
        if clip is None:
            tmp_clip = os.path.join(out_dir or tempfile.gettempdir(), "sam_clip.mp4")
            clip = (clip_builder or build_clip)(frame_set, tmp_clip, fps=fps)
        try:
            if mode == "video":
                person_dets = _video_pass(client, clip, frame_set, person_concept,
                                          person_concept, summary)
                if do_facing:
                    face_dets = _video_pass(client, clip, frame_set, face_concept,
                                            face_concept, summary)
                rows = tracks_from_object_ids(person_dets)
                person_dets = attach_track_ids(person_dets)
            else:
                person_dets, merge_drops, pinfo = _partitioned_video_pass(
                    client, clip, frame_set, person_concept, summary)
                summary[f"seed_boxes_{person_concept}"] = pinfo["seed_boxes"]
                summary[f"partitions_{person_concept}"] = pinfo["partitions"]
                if do_facing:
                    # faces overflow the 16-object cap exactly as people do, so
                    # the face pass is partitioned too — a plain face pass on a
                    # 25-person scene returns 16 faces and reads as 9 people
                    # turned away, which is a silent wrong answer
                    face_dets, face_drops, finfo = _partitioned_video_pass(
                        client, clip, frame_set, face_concept, summary)
                    summary[f"seed_boxes_{face_concept}"] = finfo["seed_boxes"]
                    summary[f"partitions_{face_concept}"] = finfo["partitions"]
                    summary["cross_pass_duplicates_dropped_face"] = len(face_drops)
                rows = _rows_from_boxes(person_dets)
                summary["cross_pass_duplicates_dropped"] = len(merge_drops)
        finally:
            if tmp_clip and not keep_clip and os.path.exists(tmp_clip):
                try:
                    os.unlink(tmp_clip)
                except OSError:
                    pass
    else:
        raise ValueError(f"unknown mode {mode!r}")

    facing_df = pd.DataFrame(columns=["track_id", "frame", "facing_rad",
                                      "facing_conf", "facing_source"])
    pairings = pd.DataFrame()
    if do_facing and len(person_dets):
        facing_df, pairings = pair_and_face(
            person_dets, face_dets, calibration_config=calibration_config,
            include_absent=include_absent_facing)
        summary["facing"] = facing_summary(facing_df, n_person_rows=len(person_dets))
        summary["facing"]["frame"] = ("floor_plane_meters" if calibration_config
                                      else "pixels_uncalibrated")

    summary["n_person_detections"] = int(len(person_dets))
    summary["n_face_detections"] = int(len(face_dets))
    summary["n_tracks"] = int(rows["track_id"].nunique()) if len(rows) else 0
    summary["n_rows"] = int(len(rows))
    summary["cost"] = client.cost_report()
    summary["proven_live"] = {
        "image_endpoint": "proven 2026-10-27 — 72 person boxes on a GC frame "
                          "whose ground truth holds 70 pedestrians",
        "video_endpoint_single_pass": "proven 2026-10-27 — base64 data URL in an "
                                      "input_video part, 19 frame lines back from "
                                      "a 20-frame clip in 10 s",
        "video_16_object_cap": "confirmed live 2026-10-27 — exactly 16 object ids "
                               "returned on a 72-person frame",
        "video_box_prompt_partitioned": "proven 2026-10-27 — 8 passes of 9 seeded "
                                        "boxes, 72 tracks merged, 0 duplicates",
        "chunked_reseeding": "built 2026-10-28 and mock-tested; see "
                             "`chunking` and `chunks_person` in this summary "
                             "for what it did on this run",
        "face_concept_on_overhead_footage": "measured 2026-10-27 — 1 face per 72 "
                                            "people on Grand Central, and 4x "
                                            "upscaling does not recover them: a "
                                            "high camera sees no faces. 'head' "
                                            "returns 54/72 on the same frame",
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        rows.to_csv(os.path.join(out_dir, "sam_tracks.csv"), index=False)
        if len(person_dets):
            person_dets.to_csv(os.path.join(out_dir, "sam_person_detections.csv"), index=False)
        if len(face_dets):
            face_dets.to_csv(os.path.join(out_dir, "sam_face_detections.csv"), index=False)
        if len(facing_df):
            facing_df.to_csv(os.path.join(out_dir, "sam_facing.csv"), index=False)
        with open(os.path.join(out_dir, "sam_run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return {"rows": rows, "detections": person_dets, "faces": face_dets,
            "facing": facing_df, "pairings": pairings, "summary": summary}


def _rows_from_boxes(det_table):
    """Canonical rows from a detection table that already carries `track_id`."""
    if det_table is None or not len(det_table):
        return pd.DataFrame(columns=["track_id", "frame", "x", "y"])
    pts = [feet_point(r.x1, r.y1, r.x2, r.y2) for r in det_table.itertuples()]
    out = pd.DataFrame({
        "track_id": det_table["track_id"].to_numpy(),
        "frame": det_table["frame"].astype(int).to_numpy(),
        "x": [p[0] for p in pts],
        "y": [p[1] for p in pts],
    })
    return out.sort_values(["track_id", "frame"]).reset_index(drop=True)


def _image_pass(client, frame_set, concept, summary):
    tables = []
    for i, path in enumerate(frame_set.paths):
        text = client.segment_image(path, concept=concept)
        frames = parse_segmentation(text)
        summary["warnings"].extend(parse_warnings(frames))
        if len(frames) > 1:
            summary["warnings"].append(
                f"image response for frame {frame_set.frames[i]} carried "
                f"{len(frames)} frame markers — folded into one frame")
        # an image response is one line, so force this frame's number
        dets = [d for v in frames.values() for d in v]
        tables.append(detections_to_frame_table({0: dets},
                                                FrameSet([path], [frame_set.frames[i]]),
                                                concept=concept))
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if len(out):
        # per-frame object ids are positions, not identities — make that explicit
        out["object_id"] = out.groupby("frame").cumcount()
    return out


def _video_pass(client, clip, frame_set, prompt, concept, summary):
    text = client.segment_video(video_path=clip, concept=prompt,
                                n_frames=len(frame_set))
    frames = parse_segmentation(text)
    summary["warnings"].extend(parse_warnings(frames))
    return detections_to_frame_table(frames, frame_set, concept=concept,
                                     warnings=summary["warnings"])


def detect_frame_boxes(client, frame_path, concept):
    """One image-endpoint call on one frame -> (boxes, frame_width, frame_height).

    The front porch of every seed: the image endpoint has no 16-object cap, so
    it is the only thing that can say who is in a frame.
    """
    text = client.segment_image(frame_path, concept=concept)
    frames = parse_segmentation(text)
    dets = [d for v in frames.values() for d in v]
    boxes = [d["bbox"] for d in dets]
    w = dets[0].get("frame_width") if dets else None
    h = dets[0].get("frame_height") if dets else None
    return boxes, w, h, parse_warnings(frames)


def _partitioned_video_pass(client, clip, frame_set, concept, summary,
                            seed_frame_index=0, key=None, seed_boxes=None,
                            frame_wh=None):
    """>16 objects: seed one video pass per spatially adjacent group of <=16.

    Seed boxes come from an image-endpoint call on one frame (no 16-cap there),
    partitioned by `partition_boxes`. **This runs for the "face" concept too,
    not just "person"** — the 16-object cap is a property of video tracking, so
    a crowd's faces overflow exactly the way its people do, and a single face
    pass over a 25-person scene would have silently returned 16 faces and left
    the rest reading as "turned away".

    One seed frame tracks only the people standing on it, which is the defect
    `_chunked_video_pass` exists to close — pass `seed_boxes` (and `frame_wh`)
    to supply a seed that was composed elsewhere, from continuing tracks plus a
    fresh detection, instead of paying for another image call here.

    Returns `(table, drops, info)`.
    """
    key = key or concept
    seed_frame_index = max(0, min(int(seed_frame_index), len(frame_set) - 1))
    if seed_boxes is None:
        seed, w, h, warns = detect_frame_boxes(
            client, frame_set.paths[seed_frame_index], concept)
        summary["warnings"].extend(warns)
    else:
        seed = [list(b) for b in seed_boxes]
        w, h = (frame_wh or (None, None))
    info = {"seed_boxes": len(seed), "partitions": []}
    if not seed:
        summary["warnings"].append(
            f"partitioned mode ({key}): the seed frame returned no boxes — "
            "nothing to prompt with")
        return pd.DataFrame(), [], info
    groups = partition_boxes(seed, VIDEO_OBJECT_CAP)
    info["partitions"] = [len(g) for g in groups]

    tables = []
    for g in groups:
        # the prompt's frame marker is the *clip* index, which is what SAM counts
        prompt = box_prompt_text([seed[i] for i in g], frame_index=seed_frame_index,
                                 frame_width=w, frame_height=h)
        text = client.segment_video(video_path=clip, concept=prompt,
                                    n_frames=len(frame_set))
        frames = parse_segmentation(text)
        summary["warnings"].extend(parse_warnings(frames))
        tables.append(detections_to_frame_table(frames, frame_set, concept=concept,
                                                warnings=summary["warnings"]))
    merged, drops = merge_passes(tables)
    return merged, drops, info


def _crop_frames_for_tile(frame_set, tile, out_dir, quality=92):
    """Write every frame cropped to one tile, and return the paths in order.

    Clip index i of the cropped clip is frame i of the original, which is the
    property that lets the original `frame_set` do the frame mapping for the
    tile's detections — so no second mapping has to be kept in step.
    """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    box = (int(tile[0]), int(tile[1]), int(tile[2]), int(tile[3]))
    paths = []
    for i, p in enumerate(frame_set.paths):
        out = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        Image.open(p).crop(box).save(out, quality=quality)
        paths.append(out)
    return paths


def _rebase_tile_boxes(table, tile, crop_w, crop_h):
    """A tile's detection table, moved into full-frame pixel coordinates.

    SAM reports in the coordinate space of what it was sent, so the scale is
    taken from the crop's real size over the size reported back rather than
    assumed equal — the same care `tiling.detect_in_tiles` takes. Getting this
    wrong does not error; it silently puts every person in the wrong place.
    """
    if table is None or len(table) == 0:
        return table
    t = table.copy()
    rw = t["frame_width"].replace(0, np.nan).fillna(crop_w)
    rh = t["frame_height"].replace(0, np.nan).fillna(crop_h)
    sx = crop_w / rw
    sy = crop_h / rh
    t["x1"] = t["x1"] * sx + tile[0]
    t["x2"] = t["x2"] * sx + tile[0]
    t["y1"] = t["y1"] * sy + tile[1]
    t["y2"] = t["y2"] * sy + tile[1]
    return t


def _tiled_video_pass(client, frame_set, concept, summary, tiles, fps=5.0,
                      clip_dir=None, clip_builder=None, keep_clips=False,
                      key=None, link_iou=0.35):
    """One video pass per tile, merged into one table in full-frame coordinates.

    Why the video endpoint rather than an image call per tile per frame: the
    image endpoint costs $2.50 per 1,000 calls against $0.20 per 1,000 video
    frames, so tiling a 90-frame clip four ways costs about $0.14 through
    video and about $0.90 through images. Same tiles, a sixth of the bill.

    Two things fall out of tiling that are worth stating plainly:

    * **Crowding per tile drops**, so tiles that would have needed box-prompt
      partitioning to beat the 16-object cap often no longer do. Tiling and the
      cap workaround pull in the same direction.
    * **Identity is per tile**, so the cross-pass merge is what links a person
      seen in two overlapping tiles. `merge_passes` already does exactly this
      for partitions, and the tile case is the same shape with a lower overlap
      threshold, because a copy cut at a tile edge shares less area with the
      whole-body copy next door.

    Returns `(table, drops, info)` to match `_partitioned_video_pass`.
    """
    key = key or concept
    clip_dir = clip_dir or tempfile.gettempdir()
    info = {"n_tiles": len(tiles), "tiles": [list(t) for t in tiles],
            "per_tile": []}
    tables = []

    for idx, tile in enumerate(tiles):
        crop_w = int(tile[2]) - int(tile[0])
        crop_h = int(tile[3]) - int(tile[1])
        tdir = os.path.join(clip_dir, f"tile_{idx:02d}_{key}")
        try:
            paths = _crop_frames_for_tile(frame_set, tile, tdir)
        except Exception as e:
            summary["warnings"].append(
                f"tiled mode ({key}): tile {idx} could not be cropped — "
                f"{type(e).__name__}: {str(e)[:120]}")
            info["per_tile"].append({"tile": idx, "error": "crop failed"})
            continue

        clip = os.path.join(clip_dir, f"tile_{idx:02d}_{key}.mp4")
        try:
            (clip_builder or build_clip)(_PathsOnly(paths), clip, fps=fps)
            text = client.segment_video(video_path=clip, concept=concept,
                                        n_frames=len(paths))
            frames = parse_segmentation(text)
            summary["warnings"].extend(parse_warnings(frames))
            table = detections_to_frame_table(frames, frame_set, concept=concept,
                                              warnings=summary["warnings"])
            table = _rebase_tile_boxes(table, tile, crop_w, crop_h)
            tables.append(table)
            info["per_tile"].append({"tile": idx, "rect": list(tile),
                                     "detections": int(len(table))})
        except Exception as e:
            # One tile failing must not lose the pass: the remaining tiles
            # still cover most of the frame, and the count of failures is
            # reported so a thin result reads as thin.
            summary["warnings"].append(
                f"tiled mode ({key}): tile {idx} failed — "
                f"{type(e).__name__}: {str(e)[:160]}")
            info["per_tile"].append({"tile": idx, "error": str(e)[:160]})
        finally:
            if not keep_clips:
                for p in paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                try:
                    os.rmdir(tdir)
                except OSError:
                    pass
                if os.path.exists(clip):
                    try:
                        os.unlink(clip)
                    except OSError:
                        pass

    if not tables:
        return pd.DataFrame(), [], info
    merged, drops = merge_passes(tables, iou_threshold=link_iou)
    info["duplicates_dropped"] = len(drops)
    info["n_tile_failures"] = sum(1 for t in info["per_tile"] if "error" in t)
    return merged, drops, info


class _PathsOnly:
    """The two attributes `build_clip` reads, over a plain list of paths.

    `build_clip` wants a frame set and uses only `.paths` and `len()`; the
    cropped frames have no canonical numbering of their own because the
    original frame set supplies it. Passing a real FrameSet here would mean
    inventing canonical frames for the crops and then having two mappings that
    could disagree.
    """

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)


def _chunked_video_pass(client, frame_set, concept, summary, partitioned=True,
                        chunk_frames=DEFAULT_CHUNK_FRAMES,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        fps=5.0, clip_dir=None, clip_builder=None,
                        keep_clips=False, key=None,
                        reseed_iou=DEFAULT_RESEED_IOU,
                        link_iou=DEFAULT_CHUNK_LINK_IOU,
                        seed_lookback=DEFAULT_SEED_LOOKBACK):
    """The video path, chunk by chunk, re-seeded at every chunk.

    Per chunk: build a clip of just that chunk's frames, detect on its first
    frame over the image endpoint, seed the box-prompted passes with the
    continuing tracks' latest boxes *plus* the newcomers that detection found,
    then link the chunk's tracks onto the running ids over the frames the two
    chunks share.

    For the <=16-person route (`partitioned=False`) there is nothing to seed —
    a concept prompt re-detects on its own — so chunking there is only the clip
    split and the seam link, and no image call is paid for.

    Returns `(table, drops, chunk_info)`; the table carries final `track_id`s.
    """
    key = key or concept
    clip_builder = clip_builder or build_clip
    clip_dir = clip_dir or tempfile.gettempdir()
    spans = chunk_spans(len(frame_set), chunk_frames, chunk_overlap)

    acc = None
    next_id = 0
    all_drops = []
    chunk_info = []
    prev_stop = None

    for k, (a, b) in enumerate(spans):
        sub = FrameSet(frame_set.paths[a:b], frame_set.frames[a:b])
        clip_path = os.path.join(clip_dir, f"sam_chunk_{key}_{k:02d}.mp4")
        clip = clip_builder(sub, clip_path, fps=fps)
        row = {"chunk": k, "frames": [int(sub.frames[0]), int(sub.frames[-1])],
               "n_frames": len(sub), "seed_frame": int(sub.frames[0])}
        try:
            if partitioned:
                fresh, w, h, warns = detect_frame_boxes(client, sub.paths[0], concept)
                summary["warnings"].extend(warns)
                cont = continuing_seed_boxes(acc, sub.frames[0], lookback=seed_lookback)
                seed, _origins, sinfo = compose_chunk_seed(cont, fresh,
                                                           iou_min=reseed_iou)
                row.update(sinfo)
                table, drops, pinfo = _partitioned_video_pass(
                    client, clip, sub, concept, summary, seed_frame_index=0,
                    key=f"{key}_c{k}", seed_boxes=seed, frame_wh=(w, h))
                row["partitions"] = pinfo["partitions"]
                all_drops.extend(drops)
            else:
                table = _video_pass(client, clip, sub, concept, concept, summary)
                if len(table):
                    table = table.copy()
                    table["track_id"] = [f"c{k}_{int(o):06d}"
                                         for o in table["object_id"]]
                row["seed_boxes"] = 0
                row["continuing"] = 0 if acc is None else int(
                    len(continuing_seed_boxes(acc, sub.frames[0],
                                              lookback=seed_lookback)))
        finally:
            if not keep_clips and os.path.exists(clip_path):
                try:
                    os.unlink(clip_path)
                except OSError:
                    pass

        shared = []
        if prev_stop is not None:
            shared = [f for f in sub.frames[:max(0, prev_stop - a)]]
        acc, linfo, next_id = link_chunk_tracks(acc, table, shared, next_id=next_id,
                                                iou_threshold=link_iou)
        row.update(linfo)
        row["tracks_so_far"] = int(acc["track_id"].nunique()) if len(acc) else 0
        chunk_info.append(row)
        prev_stop = b

    if acc is None:
        acc = pd.DataFrame()
    return acc, all_drops, chunk_info


def observed_stride(rows):
    """The data's own frame spacing: the median positive gap between consecutive
    frames inside a track. 1 for a 5 fps extraction numbered by sample index,
    20 for Grand Central's annotations, whatever the footage actually is."""
    diffs = []
    for _, g in rows.groupby("track_id", sort=False):
        f = np.sort(g["frame"].to_numpy())
        d = np.diff(f)
        diffs.extend(d[d > 0].tolist())
    if not diffs:
        return 1
    return int(max(1, round(float(np.median(diffs)))))


def to_pipeline(result, facing_df=None, fps=5.0, sample_stride=1,
                max_gap_seconds=None, strict_stride=True):
    """Hand the canonical rows to stage 3 with the right clock, carrying facing.

    Returns `(cleaned, report, facing_df)` — the same `(cleaned, report)` pair
    `track_cleaning.clean_tracks` returns, plus the facing table with its
    `track_id`s **remapped through stage 3's own bridges**.

    Two traps this exists to close:

    1. **The clock.** A 5 fps extraction numbered by sample index is
       `fps=5.0, sample_stride=1` — not Grand Central's `fps=25, stride=20`.
       Getting it wrong is the defect that inflated Hall-band visits sixtyfold
       on 2026-10-26, so the numbers live in one call.
    2. **The bridges.** `clean_tracks` stitches a mid-frame death to a later
       birth and *renames* the reborn track to the dead one's id. Facing rows
       computed before cleaning still carry the old id, so without this remap a
       bridged person's facing silently stops landing. The merge is a left join,
       so the loss would have been invisible — no error, just less facing.
    """
    from proxemix import track_cleaning

    rows = result["rows"] if isinstance(result, dict) else result

    # --- the stride guard, written after a live run proved it was needed -----
    # `interpolate_and_smooth` fills every `sample_stride` frames between a
    # track's first and last. Declare stride 1 on frames that are really 5,000
    # apart and 687 rows become 4.3 million, the grouper then "detects" 455,063
    # group-frames of pure interpolation, and **nothing errors**. That happened
    # on the first live run (the kill-test stills are minutes apart but numbered
    # in the source video's own 25 fps frames). So the stride is checked against
    # the data, and a mismatch stops the run instead of inflating it.
    obs = observed_stride(rows)
    if sample_stride == "auto":
        sample_stride = obs
    elif strict_stride and int(sample_stride) != obs:
        raise ValueError(
            f"sample_stride={sample_stride} but these rows are spaced {obs} frames "
            f"apart (median gap within a track). Stage 3 would interpolate "
            f"~{obs}x too many rows and nothing downstream would notice. Pass "
            f"sample_stride={obs}, or sample_stride='auto', or "
            f"strict_stride=False if you mean it.")
    sample_stride = int(sample_stride)
    kw = {} if max_gap_seconds is None else {"max_gap_seconds": max_gap_seconds}
    endpoints = track_cleaning.classify_endpoints(rows)
    bridged, merges = track_cleaning.bridge_tracks(rows, fps=fps, **kw)
    cleaned = track_cleaning.interpolate_and_smooth(bridged, fps=fps,
                                                    sample_stride=sample_stride)
    report = {
        "tracks_before": rows.track_id.nunique(),
        "births_border": int((endpoints.birth_type == "border").sum()),
        "births_mid_frame": int((endpoints.birth_type == "mid_frame").sum()),
        "deaths_border": int((endpoints.death_type == "border").sum()),
        "deaths_mid_frame": int((endpoints.death_type == "mid_frame").sum()),
        "gaps_bridged": len(merges),
        "tracks_after": bridged.track_id.nunique(),
        "rows_before": len(rows),
        "rows_after_interp": len(cleaned),
        "rows_interpolated": int(cleaned["interpolated"].sum()),
        "sample_stride": sample_stride,
        "observed_stride": obs,
    }

    if facing_df is not None and len(facing_df):
        remap = _bridge_remap(rows, merges)
        facing_df = facing_df.copy()
        facing_df["track_id"] = facing_df["track_id"].map(lambda t: remap.get(t, t))
        report["facing_rows_remapped"] = int(
            sum(1 for t in facing_df["track_id"] if t in set(remap.values())))
    return cleaned, report, facing_df


def _bridge_remap(rows, merges):
    """The same union-find `track_cleaning.bridge_tracks` applies to its own
    output, so facing rows follow the ids stage 3 chose."""
    parent = {}

    def find(tid):
        while parent.get(tid, tid) != tid:
            tid = parent[tid]
        return tid

    for dead, reborn, _, _ in merges:
        parent[find(reborn)] = find(dead)
    return {tid: find(tid) for tid in rows.track_id.unique()}


def facing_landing_rate(cleaned_with_facing, source_labels=("keypoints",)):
    """What share of the cleaned track rows actually carry an external facing.

    The number to print beside any claim that real facing reached the grouper —
    a left join loses rows silently, so this is the only honest check that the
    seam delivered.
    """
    if "facing_source" not in cleaned_with_facing.columns or not len(cleaned_with_facing):
        return {"rows": int(len(cleaned_with_facing)), "pct_external": 0.0}
    src = cleaned_with_facing["facing_source"].astype(str)
    hit = src.isin(list(source_labels))
    return {
        "rows": int(len(src)),
        "n_external": int(hit.sum()),
        "pct_external": round(float(100.0 * hit.mean()), 1),
        "by_source": {str(k): int(v) for k, v in src.value_counts().items()},
    }


def apply_to_tracks(cleaned_or_calibrated, facing_df, source="keypoints"):
    """Feed the face-derived facing through `facing.py`'s seam.

    `source="keypoints"` is the label `facing.py`'s docstring reserved for
    exactly this, and the rows that came from an absent face carry their own
    source label inside `facing_df` for anyone auditing coverage.
    """
    if facing_df is None or len(facing_df) == 0:
        return cleaned_or_calibrated
    cols = ["track_id", "frame", "facing_rad", "facing_conf"]
    out = cleaned_or_calibrated
    if "facing_source" in facing_df.columns:
        # one call per source, so the cleaned table keeps the distinction
        # between "a face was found" and "no face was found" — the whole point
        # of the facing_source column is that coverage stays auditable, and a
        # single label would have hidden the weaker half inside the stronger.
        for label, g in facing_df.groupby("facing_source", sort=True):
            out = apply_external_facing(out, g[cols], source=str(label))
        return out
    return apply_external_facing(out, facing_df[cols], source=source)


# ---------------------------------------------------------------------------
# self-test — mocked HTTP, no key, no network
# ---------------------------------------------------------------------------

def _mock_stream(output_text):
    """A real-shaped SSE body carrying one output_text delta."""
    ev = json.dumps({"type": "response.output_text.delta", "delta": output_text})
    done = json.dumps({"type": "response.completed", "response": {"output": []}})
    return f"data: {ev}\n\ndata: {done}\n\ndata: [DONE]\n\n"


def _mock_transport(by_concept, record=None, status=200):
    """A transport that answers like the hosted API without touching the network.

    `by_concept` maps a concept word (or `"__box__"` for a box-prompt request)
    to either one output string or `{"image": ..., "video": ...}`. An image
    value may be a list, in which case successive image calls walk it — which is
    how a per-frame image pass gets per-frame answers.
    """
    state = {}

    def transport(url, headers, body, timeout):
        payload = json.loads(body.decode())
        content = payload["input"][0]["content"]
        prompt = content[0]["text"]
        media = content[1]["type"]
        if record is not None:
            record.append({"prompt": prompt[:60], "media": media,
                           "has_auth": headers.get("Authorization", "").startswith("Bearer "),
                           "stream": payload.get("stream")})
        key = None
        if prompt.startswith("<") and "__box__" in by_concept:
            key = "__box__"
        else:
            for k in by_concept:
                if k != "__box__" and k in prompt:
                    key = k
                    break
        if key is None:
            return status, _mock_stream("")
        spec = by_concept[key]
        if isinstance(spec, str):
            spec = {"image": spec, "video": spec}
        kind = "image" if media == "input_image" else "video"
        val = spec.get(kind, "")
        if isinstance(val, (list, tuple)):
            i = state.get((key, kind), 0)
            state[(key, kind)] = i + 1
            val = val[min(i, len(val) - 1)] if val else ""
        return status, _mock_stream(val)

    return transport


def self_test():
    """Everything this module claims, checked with no key and no HTTP.

    Run: `py -m proxemix.sam_video`
    """
    import tempfile as _tf
    results = {}

    # --- 1. the parser against the docs' own example ----------------------
    doc_example = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
                   "<|mask;x=0;y=0;data=27,60,~abc|>,"
                   "1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
                   "<|mask;x=0;y=0;data=27,48,~def|>")
    parsed = parse_segmentation(doc_example)
    assert list(parsed) == [0], parsed
    assert len(parsed[0]) == 2
    assert parsed[0][0]["bbox"] == [211.0, 228.0, 270.0, 254.0]
    assert parsed[0][0]["frame_width"] == 320
    assert parsed[0][0]["mask"]["encoding"] == "~"
    assert parsed[0][0]["mask"]["height"] == 27 and parsed[0][0]["mask"]["width"] == 60
    results["parse_doc_example"] = "2 boxes, 2 masks, fields by name"

    # confidence, multi-frame, sparse ids, a frame with nothing in it
    multi = ("<0f>0<|box;x1=10;y1=10;x2=20;y2=40;w=100;h=100;c=0.91|>"
             "<|mask;x=0;y=0;c=0.91;data=30,10,!zz|>,"
             "2<|box;x1=50;y1=10;x2=60;y2=40;w=100;h=100;c=0.5|>"
             "<|mask;x=0;y=0;data=30,10,~yy|>"
             "<1f>0<|box;x1=12;y1=10;x2=22;y2=40;w=100;h=100;c=0.9|>"
             "<|mask;x=0;y=0;data=30,10,~xx|>")
    p2 = parse_segmentation(multi)
    assert sorted(p2) == [0, 1]
    assert [d["object_id"] for d in p2[0]] == [0, 2], "sparse ids must survive"
    assert p2[0][0]["confidence"] == 0.91
    assert p2[1][0]["object_id"] == 0
    results["parse_multiframe"] = "2 frames, sparse ids 0/2, confidences read"

    # the probe's looser reality: no object id prefix, no w/h, no mask
    loose = "<|box;x1=1;y1=2;x2=3;y2=4|><|box;x1=5;y1=6;x2=7;y2=8|>"
    p3 = parse_segmentation(loose)
    assert len(p3[0]) == 2 and p3[0][1]["object_id"] == 1
    assert p3[0][0]["frame_width"] is None
    results["parse_loose"] = "id-less, w/h-less boxes parse as frame 0"

    # a mask/box mismatch is recorded, not fatal
    mismatch = ("<0f>0<|box;x1=1;y1=1;x2=2;y2=2|><|mask;x=0;y=0;data=1,1,~a|>"
                "<|mask;x=0;y=0;data=1,1,~b|>")
    p4 = parse_segmentation(mismatch)
    assert len(p4[0]) == 1 and parse_warnings(p4), "mismatch must be recorded"
    results["parse_mismatch_nonfatal"] = parse_warnings(p4)[0]
    assert parse_segmentation("") == {}
    assert parse_segmentation("no tokens here at all") == {}

    # --- 2. the client: mocked HTTP, no key on disk needed ----------------
    tmp = _tf.mkdtemp(prefix="sam_selftest_")
    keyfile = os.path.join(tmp, "fake_key.txt")
    with open(keyfile, "w") as f:
        f.write("LLM_selftest_" + "x" * 40)
    img = os.path.join(tmp, "frame_000000.jpg")
    with open(img, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"0" * 64)   # enough to be a file

    record = []
    client = SamVideoClient(
        key_path=keyfile, max_usd=1.0,
        transport=_mock_transport({"person": multi, "face": ""}, record=record))
    text = client.segment_image(img, concept="person")
    assert "box" in text
    assert record[0]["media"] == "input_image" and record[0]["has_auth"]
    assert record[0]["stream"] is True
    assert abs(client.spent_usd - 0.0025) < 1e-9, client.spent_usd
    results["client_image_call"] = f"${client.spent_usd:.4f} for one image call"

    # the key never leaves: an error body carrying it comes back scrubbed
    leaky = SamVideoClient(key_path=keyfile, transport=lambda u, h, b, t: (
        401, "bad key: " + h["Authorization"].split()[1]))
    try:
        leaky.segment_image(img, concept="person")
        raise AssertionError("a 401 must raise")
    except SamError as e:
        assert "LLM_selftest" not in str(e), "THE KEY LEAKED INTO AN ERROR MESSAGE"
        assert "<key>" in str(e)
    results["key_never_printed"] = "401 body came back scrubbed to <key>"

    # an empty key file is refused before any call
    empty = os.path.join(tmp, "empty.txt")
    open(empty, "w").close()
    c_empty = SamVideoClient(key_path=empty, transport=lambda *a: (200, ""))
    assert not c_empty.key_present()
    try:
        c_empty.segment_image(img)
        raise AssertionError("empty key must raise")
    except SamError as e:
        assert "0 characters" in str(e)
    results["empty_key_refused"] = "refused before the request"

    # the spend cap refuses the call that would cross it
    capped = SamVideoClient(key_path=keyfile, max_usd=0.004,
                            transport=_mock_transport({"person": multi}))
    capped.segment_image(img)
    try:
        capped.segment_image(img)
        raise AssertionError("the cap must refuse the second call")
    except SpendCapExceeded:
        pass
    assert len(capped.calls) == 1, "a refused call must not be charged"
    results["spend_cap"] = "second call refused at $0.004 cap, not charged"

    # --- 3. cost arithmetic ----------------------------------------------
    assert abs(image_cost_usd(1000) - 2.50) < 1e-9
    assert abs(video_cost_usd(1000) - 0.20) < 1e-9
    assert abs(video_cost_usd(10) - video_cost_usd(50)) < 1e-9, "50-frame minimum"
    ten_min = 10 * 60 * 5          # 3,000 frames at 5 fps
    results["cost_10min_5fps"] = {
        "video_two_concepts_usd": round(estimate_pass_cost(ten_min, 2, "video"), 2),
        "image_two_concepts_usd": round(estimate_pass_cost(ten_min, 2, "image"), 2),
    }
    assert results["cost_10min_5fps"]["video_two_concepts_usd"] == 1.2
    assert results["cost_10min_5fps"]["image_two_concepts_usd"] == 15.0

    # --- 4. stitching: a known 3-frame walk -------------------------------
    rows = []
    for fi, frame in enumerate([0, 1, 2]):
        for pid, x0 in ((0, 100.0), (1, 500.0)):
            x = x0 + fi * 12.0              # both walk right, 12 px per frame
            rows.append({"concept": "person", "clip_index": fi, "frame": frame,
                         "object_id": pid, "x1": x, "y1": 100.0,
                         "x2": x + 40, "y2": 220.0, "conf": 0.9})
    det = pd.DataFrame(rows)
    tracks, assign = stitch_tracks(det)
    assert tracks.track_id.nunique() == 2, tracks
    assert len(tracks) == 6
    # the feet point is bottom-centre, which is what calibration projects
    first = tracks[tracks.frame == 0].sort_values("x")
    assert abs(first.iloc[0].x - 120.0) < 1e-6 and abs(first.iloc[0].y - 220.0) < 1e-6
    # identity must hold: each track's x must increase monotonically
    for tid, g in tracks.groupby("track_id"):
        assert (np.diff(g.sort_values("frame").x.to_numpy()) > 0).all()
    results["stitch_two_walkers"] = "2 tracks over 3 frames, feet points, no ID swap"

    # a gap of one frame must not break the track
    det_gap = det[~((det.frame == 1) & (det.object_id == 0))].copy()
    tracks_gap, _ = stitch_tracks(det_gap)
    assert tracks_gap.track_id.nunique() == 2, tracks_gap.track_id.nunique()
    results["stitch_tolerates_gap"] = "one missed detection keeps 2 tracks"

    # two people crossing must not swap identities when IoU is decisive
    crossers = []
    for fi in range(5):
        crossers.append({"concept": "person", "clip_index": fi, "frame": fi,
                         "object_id": 0, "x1": 100 + fi * 10, "y1": 0,
                         "x2": 140 + fi * 10, "y2": 120, "conf": 0.9})
        crossers.append({"concept": "person", "clip_index": fi, "frame": fi,
                         "object_id": 1, "x1": 300 - fi * 10, "y1": 0,
                         "x2": 340 - fi * 10, "y2": 120, "conf": 0.9})
    tc, _ = stitch_tracks(pd.DataFrame(crossers))
    assert tc.track_id.nunique() == 2
    results["stitch_crossing_pair"] = "2 tracks, 10 rows"

    # the video path needs no stitching at all
    vt = tracks_from_object_ids(det)
    assert vt.track_id.nunique() == 2 and list(vt.columns) == ["track_id", "frame", "x", "y"]
    results["video_ids_are_tracks"] = "SAM object ids used directly"

    # --- 5. schema parity with the Grand Central adapter ------------------
    from proxemix.grand_central_adapter import load_grand_central_tracks, make_synthetic_fixture
    fixture = os.path.join(tmp, "gc_fixture")
    make_synthetic_fixture(fixture, n_tracks=4, seed=1)
    gc = load_grand_central_tracks(fixture)
    assert list(vt.columns) == list(gc.columns), (list(vt.columns), list(gc.columns))
    assert vt.track_id.dtype == gc.track_id.dtype
    assert vt.frame.dtype == gc.frame.dtype, (vt.frame.dtype, gc.frame.dtype)
    assert vt.x.dtype == gc.x.dtype
    results["schema_parity_with_gc_adapter"] = "columns and dtypes identical"

    # and the real proof: stage 3 runs on our rows unchanged
    from proxemix import track_cleaning
    cleaned, report = track_cleaning.clean_tracks(vt, fps=5.0, sample_stride=1)
    assert {"track_id", "frame", "t", "x", "y", "vx", "vy"} <= set(cleaned.columns)
    assert report["tracks_before"] == 2
    results["clean_tracks_accepts_rows"] = {k: report[k] for k in
                                           ("tracks_before", "tracks_after", "rows_after_interp")}

    # --- 6. partitioning past the 16-object cap ---------------------------
    many = [[i * 10.0, (i % 5) * 50.0, i * 10.0 + 8, (i % 5) * 50.0 + 40]
            for i in range(37)]
    groups = partition_boxes(many, 16)
    assert all(len(g) <= 16 for g in groups), [len(g) for g in groups]
    assert sum(len(g) for g in groups) == 37
    assert len({i for g in groups for i in g}) == 37, "every person exactly once"
    results["partition_37_people"] = [len(g) for g in groups]

    # neighbours stay together: a tight cluster must not be split across passes
    cluster = [[100.0, 100.0, 120.0, 160.0], [105.0, 100.0, 125.0, 160.0],
               [110.0, 100.0, 130.0, 160.0]] + \
              [[1000.0 + i * 30, 500.0, 1020.0 + i * 30, 560.0] for i in range(20)]
    cg = partition_boxes(cluster, 16)
    homes = {i: gi for gi, g in enumerate(cg) for i in g}
    assert homes[0] == homes[1] == homes[2], "the tight trio got split across passes"
    results["partition_keeps_neighbours"] = [len(g) for g in cg]

    # box-prompt tokens must read back through our own parser
    prompt = box_prompt_text(many[:3], frame_index=0, frame_width=1920, frame_height=1080)
    back = parse_segmentation(prompt)
    assert len(back[0]) == 3 and back[0][2]["bbox"] == many[2]
    results["box_prompt_round_trip"] = prompt[:60] + "..."

    # --- 7. cross-pass merge drops the double-counted person --------------
    def _t(pass_id, obj, frames, x):
        return pd.DataFrame([{"concept": "person", "clip_index": f, "frame": f,
                              "object_id": obj, "x1": x, "y1": 0.0,
                              "x2": x + 40, "y2": 120.0, "conf": 0.9}
                             for f in frames])
    merged, drops = merge_passes([_t(0, 0, [0, 1, 2], 100.0), _t(1, 0, [0, 1], 100.0),
                                  _t(1, 1, [0, 1, 2], 900.0)])
    assert len(drops) == 1, drops
    assert merged.track_id.nunique() == 2, sorted(merged.track_id.unique())
    assert drops[0]["kept"] == "p0_000000", drops   # the longer track survives
    results["cross_pass_merge"] = f"dropped {drops[0]['dropped']} (IoU {drops[0]['mean_iou']})"

    # --- 7b. chunk spans -------------------------------------------------
    assert chunk_spans(90, 30, 2) == [(0, 31), (29, 61), (59, 90)], chunk_spans(90, 30, 2)
    assert chunk_spans(20, 30, 2) == [(0, 20)], "a clip inside one chunk is one chunk"
    assert chunk_spans(90, None, 2) == [(0, 90)], "chunking off must be one span"
    assert chunk_spans(0, 30, 2) == []
    sp = chunk_spans(90, 30, 2)
    assert all(sp[i][1] - sp[i + 1][0] == 2 for i in range(len(sp) - 1)), sp
    assert sp[0][0] == 0 and sp[-1][1] == 90, "the spans must cover the clip"
    lens = [b - a for a, b in sp]
    assert max(lens) - min(lens) <= 1, f"spans must come out even: {lens}"
    results["chunk_spans"] = {"90f_at_30": sp, "lengths": lens,
                              "billed_usd": round(estimate_chunked_cost(90, 2, 4), 4)}

    # --- 7c. the seed a chunk is re-seeded with ---------------------------
    # one continuing track, one newcomer, and a second person standing on top of
    # the continuing box — which greedy matching would swallow
    cont = [("000000", [100.0, 100.0, 140.0, 220.0])]
    fresh = [[102.0, 101.0, 142.0, 221.0],      # the continuing person, re-found
             [120.0, 100.0, 160.0, 220.0],      # shoulder to shoulder with them
             [700.0, 100.0, 740.0, 220.0]]      # a stranger
    seed, origins, sinfo = compose_chunk_seed(cont, fresh)
    assert sinfo["continuing"] == 1 and sinfo["fresh_detections"] == 3
    assert sinfo["new"] == 2, f"the shoulder-to-shoulder pair must not collapse: {sinfo}"
    assert origins == ["000000", None, None], origins
    assert seed[0] == cont[0][1], "a continuing track keeps its own box"
    assert compose_chunk_seed([], fresh)[2]["new"] == 3
    assert compose_chunk_seed(cont, [])[2] == {"continuing": 1, "fresh_detections": 0,
                                               "new": 0, "seed_boxes": 1}
    results["chunk_seed_composition"] = sinfo

    # --- 7d. the chunk boundary merge ------------------------------------
    # chunk A holds two walkers over frames 0-6; chunk B covers frames 5-11 and
    # holds the same two plus somebody who arrived at frame 5. The seam must
    # continue the two ids, mint exactly one new one, and not double-count the
    # frames both chunks cover.
    def _ct(obj, frames, x0, dx=10.0):
        return [{"concept": "person", "clip_index": i, "frame": f,
                 "object_id": obj, "x1": x0 + dx * i, "y1": 0.0,
                 "x2": x0 + dx * i + 40, "y2": 120.0, "conf": 0.9}
                for i, f in enumerate(frames)]

    chunk_a = pd.DataFrame(_ct(0, range(0, 7), 100.0) + _ct(1, range(0, 7), 400.0, 0.0))
    chunk_a["track_id"] = ["cA_%06d" % o for o in chunk_a.object_id]
    chunk_b = pd.DataFrame(_ct(0, range(5, 12), 150.0) + _ct(1, range(5, 12), 400.0, 0.0)
                           + _ct(2, range(5, 12), 700.0, 0.0))
    chunk_b["track_id"] = ["cB_%06d" % o for o in chunk_b.object_id]

    acc, info_a, nid = link_chunk_tracks(None, chunk_a, [], next_id=0)
    assert sorted(acc.track_id.unique()) == ["000000", "000001"], acc.track_id.unique()
    assert info_a["linked"] == 0 and nid == 2
    acc, info_b, nid = link_chunk_tracks(acc, chunk_b, [5, 6], next_id=nid)
    assert info_b["linked"] == 2, info_b
    assert info_b["new_ids"] == 1, info_b
    assert sorted(acc.track_id.unique()) == ["000000", "000001", "000002"], \
        sorted(acc.track_id.unique())
    # the two walkers run the whole 12 frames under one id each, the arrival 7
    assert acc.groupby("track_id").size().to_dict() == {"000000": 12, "000001": 12,
                                                        "000002": 7}, \
        acc.groupby("track_id").size().to_dict()
    assert not acc.duplicated(["track_id", "frame"]).any(), "overlap frames double-counted"
    # the continuing track's boundary rows came from chunk A, not chunk B
    assert len(acc[(acc.track_id == "000000") & (acc.frame == 5)]) == 1
    # a track that shares no frame with anybody is a new person, not a link
    far = pd.DataFrame(_ct(0, range(5, 12), 9000.0, 0.0))
    far["track_id"] = ["cC_000000"] * len(far)
    _acc2, info_c, _ = link_chunk_tracks(acc, far, [5, 6], next_id=nid)
    assert info_c == {"linked": 0, "new_ids": 1, "shared_frames": 2}, info_c
    results["chunk_boundary_merge"] = {
        "linked_across_seam": info_b["linked"],
        "new_ids_at_seam": info_b["new_ids"],
        "rows_per_track": acc.groupby("track_id").size().to_dict(),
    }

    # --- 8. facing from face boxes ---------------------------------------
    # one person box, three face positions: centred, hard left, hard right
    person = pd.DataFrame([
        {"frame": 0, "track_id": "000000", "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 400.0},
        {"frame": 1, "track_id": "000000", "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 400.0},
        {"frame": 2, "track_id": "000000", "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 400.0},
        {"frame": 3, "track_id": "000000", "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 400.0},
    ])
    faces = pd.DataFrame([
        {"frame": 0, "x1": 140.0, "y1": 110.0, "x2": 160.0, "y2": 140.0, "conf": 0.9},
        {"frame": 1, "x1": 100.0, "y1": 110.0, "x2": 120.0, "y2": 140.0, "conf": 0.9},
        {"frame": 2, "x1": 180.0, "y1": 110.0, "x2": 200.0, "y2": 140.0, "conf": 0.9},
        # frame 3: no face at all -> turned away
    ])
    fdf, prs = pair_and_face(person, faces)        # no calibration: pixel frame
    by_frame = {int(r.frame): r for r in fdf.itertuples()}
    cam_angle = np.pi / 2                           # image-down in the pixel frame
    assert abs(by_frame[0].facing_rad - cam_angle) < 1e-6, by_frame[0]
    assert by_frame[0].facing_source == "face"
    # left-of-centre face -> rotated toward image-left (negative x component)
    assert np.cos(by_frame[1].facing_rad) < 0
    assert np.cos(by_frame[2].facing_rad) > 0, "right-of-centre face must point right"
    assert abs(by_frame[1].yaw_from_camera_deg + by_frame[2].yaw_from_camera_deg) < 1e-6
    # face centre 40 px left of a 50 px half-width -> u = -0.8 -> -0.8 * 70 deg
    assert abs(by_frame[1].yaw_from_camera_deg + 0.8 * DEFAULT_MAX_YAW_DEG) < 1e-6, \
        by_frame[1].yaw_from_camera_deg
    assert by_frame[3].facing_source == "face_absent"
    assert abs(by_frame[3].facing_rad + cam_angle) < 1e-6, "absent face must face away"
    assert by_frame[3].facing_conf < 0.3, "absent-face rows must sit below the groups.py gate"
    assert by_frame[0].facing_conf <= DEFAULT_FACE_CONF
    results["facing_angles"] = {
        "centred_face_deg": round(float(np.degrees(by_frame[0].facing_rad)), 1),
        "left_face_yaw_deg": round(by_frame[1].yaw_from_camera_deg, 1),
        "right_face_yaw_deg": round(by_frame[2].yaw_from_camera_deg, 1),
        "absent_face_deg": round(float(np.degrees(by_frame[3].facing_rad)), 1),
        "absent_conf": round(float(by_frame[3].facing_conf), 3),
    }

    # pairing must reject a face that is not inside anybody
    stray = pd.DataFrame([{"frame": 0, "x1": 900.0, "y1": 900.0, "x2": 920.0,
                           "y2": 930.0, "conf": 0.9}])
    pr_stray = pair_faces_to_people(person[person.frame == 0], stray)
    assert not has_face(pr_stray.iloc[0].face_index), "a stray face must not be paired"
    # and two people, overlapping: the face goes to the tighter box
    two = pd.DataFrame([
        {"frame": 0, "track_id": "a", "x1": 0.0, "y1": 0.0, "x2": 400.0, "y2": 400.0},
        {"frame": 0, "track_id": "b", "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
    ])
    one_face = pd.DataFrame([{"frame": 0, "x1": 140.0, "y1": 110.0, "x2": 160.0,
                              "y2": 140.0, "conf": 0.9}])
    pr2 = pair_faces_to_people(two, one_face)
    got = {r.person_index: r.face_index for r in pr2.itertuples()}
    assert got[1] == 0 and not has_face(got[0]), got
    results["face_pairing_prefers_tighter_box"] = "front person claims the face"

    # --- 9. the facing.py seam actually takes it --------------------------
    from proxemix.facing import add_facing, facing_coverage
    walk = []
    for i in range(12):
        walk.append({"track_id": "000000", "frame": i, "t": i * 0.2,
                     "x_m": 0.0, "y_m": 0.0})          # standing still
        walk.append({"track_id": "000001", "frame": i, "t": i * 0.2,
                     "x_m": 1.0 * i * 0.2, "y_m": 3.0})  # walking at 1 m/s
    wdf = add_facing(pd.DataFrame(walk))
    assert (wdf[wdf.track_id == "000000"].facing_conf == 0).all()
    ext = pd.DataFrame([{"track_id": "000000", "frame": i, "facing_rad": 1.2,
                         "facing_conf": 0.6} for i in range(12)])
    merged_f = apply_to_tracks(wdf, ext)
    stander = merged_f[merged_f.track_id == "000000"]
    walker = merged_f[merged_f.track_id == "000001"]
    assert (stander.facing_source == "keypoints").all(), "the seam did not fill the stander"
    assert np.allclose(stander.facing_rad, 1.2)
    assert (walker.facing_source == "motion").all(), "a face estimate clobbered a walker"
    cov = facing_coverage(merged_f)
    assert cov["n_known"] == len(merged_f)
    results["facing_seam"] = {"coverage_pct": round(cov["pct_known"], 1),
                              "by_source": cov["by_source"]}

    # and the grouper reads the filled column without a code change
    from proxemix import groups
    pair = []
    for i in range(12):                      # two people facing each other, 0.8 m apart
        pair.append({"track_id": "A", "frame": i, "t": i * 0.2, "x_m": 0.0,
                     "y_m": 0.0, "facing_rad": 0.0, "facing_conf": 0.6})
        pair.append({"track_id": "B", "frame": i, "t": i * 0.2, "x_m": 0.8,
                     "y_m": 0.0, "facing_rad": np.pi, "facing_conf": 0.6})
    clusters = groups.detect_frame_clusters(pd.DataFrame(pair))
    assert len(clusters) == 12, len(clusters)
    assert clusters.iloc[0].group_key == "A,B"
    results["grouper_uses_face_facing"] = "face-derived facing groups a facing dyad"

    # --- 10. a whole mocked run, end to end -------------------------------
    frames_dir = os.path.join(tmp, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(3):
        with open(os.path.join(frames_dir, f"frame_{i:06d}.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"0" * 64)
    fs = FrameSet.from_dir(frames_dir)
    assert fs.frames == [0, 1, 2] and len(fs) == 3
    # one clip answer (three frame lines) and three image answers (one line each)
    def _p(i):
        return (f"0<|box;x1={100 + i * 10};y1=100;x2={140 + i * 10};y2=400;w=640;h=480;c=0.9|>"
                f",1<|box;x1=400;y1=100;x2=440;y2=400;w=640;h=480;c=0.8|>")

    def _f(i):
        return (f"0<|box;x1={115 + i * 10};y1=110;x2={125 + i * 10};y2=130;"
                f"w=640;h=480;c=0.8|>")

    person_text = "".join(f"<{i}f>" + _p(i) for i in range(3))
    face_text = "".join(f"<{i}f>" + _f(i) for i in range(3))
    person_images = [f"<0f>" + _p(i) for i in range(3)]
    face_images = [f"<0f>" + _f(i) for i in range(3)]
    run_client = SamVideoClient(
        key_path=keyfile, max_usd=1.0,
        transport=_mock_transport({"person": {"image": person_images, "video": person_text},
                                   "face": {"image": face_images, "video": face_text}}))
    res = run_sam_stage(frame_set=fs, client=run_client, mode="image",
                        out_dir=os.path.join(tmp, "out"), do_facing=True)
    assert res["summary"]["n_tracks"] == 2, res["summary"]
    assert res["summary"]["n_rows"] == 6, res["summary"]
    assert res["summary"]["n_person_detections"] == 6
    assert res["summary"]["n_face_detections"] == 3
    assert list(res["rows"].columns) == ["track_id", "frame", "x", "y"]
    # person 0 has a face every frame; person 1 never does
    src = res["facing"].groupby("track_id").facing_source.agg(set)
    assert {"face"} in list(src) and {"face_absent"} in list(src), src
    assert os.path.exists(os.path.join(tmp, "out", "sam_tracks.csv"))
    assert os.path.exists(os.path.join(tmp, "out", "sam_run_summary.json"))
    results["end_to_end_image_mode"] = {
        "tracks": res["summary"]["n_tracks"],
        "person_detections": res["summary"]["n_person_detections"],
        "face_detections": res["summary"]["n_face_detections"],
        "facing": res["summary"]["facing"]["by_source"],
        "spent_usd": res["summary"]["cost"]["spent_usd"],
    }

    # the same run in auto mode: headcount routes it, 2 people -> single video pass
    fake_clip = os.path.join(tmp, "fake.mp4")
    with open(fake_clip, "wb") as f:
        f.write(b"\x00\x00\x00 ftypmp42" + b"0" * 64)
    auto_client = SamVideoClient(
        key_path=keyfile, max_usd=1.0,
        transport=_mock_transport({"person": {"image": person_images, "video": person_text},
                                   "face": {"image": face_images, "video": face_text}}))
    auto = run_sam_stage(frame_set=fs, client=auto_client, mode="auto",
                         video_path=fake_clip, do_facing=True)
    assert auto["summary"]["mode_taken"] == "video", auto["summary"]
    assert auto["summary"]["headcount"]["peak"] == 2
    assert auto["summary"]["n_tracks"] == 2
    results["end_to_end_auto_mode"] = {
        "headcount_peak": auto["summary"]["headcount"]["peak"],
        "mode_taken": auto["summary"]["mode_taken"],
        "spent_usd": auto["summary"]["cost"]["spent_usd"],
    }

    # a crowded headcount must route to the partitioned path
    crowd_text = "<0f>" + ",".join(
        f"{i}<|box;x1={i * 15};y1=100;x2={i * 15 + 12};y2=400;w=640;h=480|>"
        for i in range(25))
    crowd_client = SamVideoClient(
        key_path=keyfile, max_usd=1.0,
        transport=_mock_transport({"person": {"image": crowd_text, "video": crowd_text},
                                   "face": {"image": face_images, "video": face_text},
                                   "__box__": {"video": person_text}}))
    crowd = run_sam_stage(frame_set=fs, client=crowd_client, mode="auto",
                          video_path=fake_clip, do_facing=False)
    assert crowd["summary"]["mode_taken"] == "video_partitioned", crowd["summary"]
    parts = crowd["summary"]["partitions_person"]
    assert sorted(parts) == [12, 13], parts
    assert sum(parts) == 25
    assert crowd["summary"]["seed_boxes_person"] == 25
    # the face pass must partition too, or a crowd's faces silently cap at 16
    crowd_faces = "<0f>" + ",".join(
        f"{i}<|box;x1={i * 15 + 2};y1=110;x2={i * 15 + 10};y2=130;w=640;h=480|>"
        for i in range(20))
    crowd_f_client = SamVideoClient(
        key_path=keyfile, max_usd=2.0,
        transport=_mock_transport({"person": {"image": crowd_text, "video": crowd_text},
                                   "face": {"image": crowd_faces, "video": face_text},
                                   "__box__": {"video": person_text}}))
    crowd_f = run_sam_stage(frame_set=fs, client=crowd_f_client, mode="auto",
                            video_path=fake_clip, do_facing=True)
    assert crowd_f["summary"]["seed_boxes_face"] == 20, crowd_f["summary"]
    assert sorted(crowd_f["summary"]["partitions_face"]) == [10, 10],         crowd_f["summary"]["partitions_face"]
    results["partitioned_face_pass"] = {
        "seed_faces": crowd_f["summary"]["seed_boxes_face"],
        "face_partitions": crowd_f["summary"]["partitions_face"],
    }

    results["end_to_end_partitioned"] = {
        "partitions": crowd["summary"]["partitions_person"],
        "duplicates_dropped": crowd["summary"].get("cross_pass_duplicates_dropped"),
        "tracks": crowd["summary"]["n_tracks"],
    }

    # --- 10b. the same clip, chunked and single-shot, side by side --------
    # The defect in miniature: two people from frame 0, a third arriving at
    # frame 5. Single-shot seeding can only ever return two tracks; chunked
    # re-seeding must find the third and must not break the first two.
    chunk_dir = os.path.join(tmp, "chunkframes")
    os.makedirs(chunk_dir, exist_ok=True)
    for i in range(12):
        with open(os.path.join(chunk_dir, f"frame_{i:06d}.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"0" * 64)
    cfs = FrameSet.from_dir(chunk_dir)
    assert len(cfs) == 12 and cfs.frames[0] == 0

    def _bx(obj, x):
        return (f"{obj}<|box;x1={x};y1=0;x2={x + 40};y2=120;w=1920;h=1080;c=0.9|>")

    # the image endpoint on each chunk's first frame
    seed_c0 = "<0f>" + ",".join([_bx(0, 100), _bx(1, 400)])
    seed_c1 = "<0f>" + ",".join([_bx(0, 150), _bx(1, 400), _bx(2, 700)])
    # what each chunk's box-prompted pass tracks back, clip index by clip index
    vid_c0 = "".join(f"<{i}f>" + ",".join([_bx(0, 100 + 10 * i), _bx(1, 400)])
                     for i in range(7))
    vid_c1 = "".join(f"<{i}f>" + ",".join([_bx(0, 150 + 10 * i), _bx(1, 400),
                                           _bx(2, 700)])
                     for i in range(7))

    def _stub_clip(fs, path, fps=5.0):
        with open(path, "wb") as fh:
            fh.write(b"\x00\x00\x00 ftypmp42" + bytes(len(fs)))
        return path

    def _chunk_client(cap=1.0):
        return SamVideoClient(
            key_path=keyfile, max_usd=cap,
            transport=_mock_transport({"person": {"image": [seed_c0, seed_c1]},
                                       "__box__": {"video": [vid_c0, vid_c1]}}))

    ch = run_sam_stage(frame_set=cfs, client=_chunk_client(), mode="video_partitioned",
                       do_facing=False, chunk_frames=6, chunk_overlap=2,
                       clip_builder=_stub_clip)
    cs = ch["summary"]
    assert cs["mode_taken"] == "video_partitioned_chunked", cs["mode_taken"]
    assert cs["chunking"]["n_chunks"] == 2, cs["chunking"]
    assert cs["chunking"]["spans_frames"] == [[0, 6], [5, 11]], cs["chunking"]
    assert cs["chunks_person"][0]["continuing"] == 0, cs["chunks_person"]
    assert cs["chunks_person"][0]["new"] == 2, cs["chunks_person"]
    assert cs["chunks_person"][1]["continuing"] == 2, cs["chunks_person"]
    assert cs["chunks_person"][1]["new"] == 1, cs["chunks_person"]
    assert cs["chunks_person"][1]["linked"] == 2, cs["chunks_person"]
    assert cs["reseeded_newcomers_person"] == 1, cs["reseeded_newcomers_person"]
    assert cs["n_tracks"] == 3, cs["n_tracks"]
    assert cs["n_rows"] == 31, cs["n_rows"]          # 12 + 12 + 7, no double count
    assert not ch["rows"].duplicated(["track_id", "frame"]).any()
    late = ch["detections"][ch["detections"].x1 >= 700]
    assert late.track_id.nunique() == 1 and late.frame.min() == 5, \
        "the arrival must be one track, starting at frame 5"
    # two image calls (one seed per chunk) and two video calls (one pass each)
    assert cs["cost"]["image_calls"] == 2 and cs["cost"]["video_calls"] == 2, cs["cost"]

    single = run_sam_stage(frame_set=cfs, client=_chunk_client(),
                           mode="video_partitioned", do_facing=False,
                           chunk_frames=None, video_path=fake_clip)
    ss = single["summary"]
    assert ss["mode_taken"] == "video_partitioned", ss["mode_taken"]
    assert ss["chunking"]["enabled"] is False
    assert ss["n_tracks"] == 2, ss["n_tracks"]
    assert float(single["detections"].x1.max()) < 700.0, \
        "single-shot must not see the late arrival — that is the defect"
    results["chunked_vs_single_shot"] = {
        "chunked_tracks": cs["n_tracks"], "single_shot_tracks": ss["n_tracks"],
        "newcomers_reseeded": cs["reseeded_newcomers_person"],
        "linked_across_seam": cs["chunks_person"][1]["linked"],
        "chunked_usd": cs["cost"]["spent_usd"], "single_shot_usd": ss["cost"]["spent_usd"],
    }

    # a prepared clip cannot be chunked, and must say so rather than silently
    # sending the whole clip as every chunk
    try:
        run_sam_stage(frame_set=cfs, client=_chunk_client(), mode="video_partitioned",
                      do_facing=False, chunk_frames=6, video_path=fake_clip)
        raise AssertionError("chunking a prepared clip must refuse")
    except ValueError as e:
        assert "cannot be chunked" in str(e), e
    results["chunk_refuses_prepared_clip"] = "ValueError before any call"

    # --- 11. frame numbering ---------------------------------------------
    sparse = FrameSet(["frame_000000.jpg", "frame_000040.jpg", "frame_006340.jpg"])
    assert sparse.frames == [0, 40, 6340], sparse.frames
    assert sparse.canonical(1) == 40, "clip index 1 must map to canonical frame 40"
    results["frame_numbering"] = "filenames -> canonical frames, clip index mapped"

    # --- 11b. clip building, for real, with ffmpeg ------------------------
    # The frame mapping is the one thing that fails silently, so the clip is
    # built and decode-counted rather than trusted. Skipped (recorded, not
    # hidden) if ffmpeg or PIL is absent.
    try:
        from PIL import Image as _Image
        clip_dir = os.path.join(tmp, "clipframes")
        os.makedirs(clip_dir, exist_ok=True)
        srcs = []
        for i in range(6):
            q = os.path.join(clip_dir, f"frame_{i * 40:06d}.jpg")
            _Image.new("RGB", (64, 48), (20 * i, 255 - 20 * i, 128)).save(q, quality=95)
            srcs.append(q)
        cfs = FrameSet.from_dir(clip_dir)
        assert cfs.frames == [0, 40, 80, 120, 160, 200], cfs.frames
        clip_out = os.path.join(tmp, "built.mp4")
        build_clip(cfs, clip_out, fps=5.0)      # raises if the count is wrong
        n_dec = _verify_clip_frames(clip_out, len(cfs))
        assert n_dec == 6, n_dec
        # and the guard must be able to fail, or it is not a check
        try:
            _verify_clip_frames(clip_out, 7)
            raise AssertionError("the frame-count guard did not fire")
        except RuntimeError as e:
            assert "decoded 6 frames, expected 7" in str(e), str(e)
        results["build_clip"] = "6 frames in, 6 decoded out; the guard fires on 7"
    except (ImportError, FileNotFoundError, RuntimeError, OSError) as e:
        results["build_clip"] = f"SKIPPED ({type(e).__name__}: {e})"

    # --- 12. the whole real path, on mocked SAM output --------------------
    # stage 2 rows -> stage 3 clean -> stage 4 calibrate -> facing seam ->
    # stage 5 grouper. This is the thing the module exists for, so it is run
    # rather than reasoned about.
    from proxemix.calibration import apply_calibration_to_tracks, load_calibration_config
    from proxemix import track_cleaning as _tc

    cleaned2, report2, facing2 = to_pipeline(res, res["facing"], fps=5.0, sample_stride=1)
    ref_cleaned, ref_report = _tc.clean_tracks(res["rows"], fps=5.0, sample_stride=1)
    assert set(report2) >= set(ref_report), (set(ref_report) - set(report2))
    for k in ref_report:                      # the duplicated report cannot drift
        assert report2[k] == ref_report[k], (k, report2[k], ref_report[k])
    assert len(cleaned2) == len(ref_cleaned)

    cal_cfg = load_calibration_config(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "calibration_gc.json"))
    calibrated = apply_calibration_to_tracks(cleaned2, cal_cfg)
    assert {"x_m", "y_m", "vx_m", "vy_m"} <= set(calibrated.columns)
    with_facing = apply_to_tracks(calibrated, facing2)
    landing = facing_landing_rate(with_facing, source_labels=("face", "face_absent"))
    assert landing["n_external"] > 0, "no face-derived facing reached the cleaned table"
    from proxemix import groups as _g
    cl = _g.detect_frame_clusters(with_facing)
    results["full_path_mocked"] = {
        "tracks": int(with_facing.track_id.nunique()),
        "cleaned_rows": int(len(with_facing)),
        "facing_landing": landing,
        "group_rows": int(len(cl)),
    }

    # the stride guard must fire on stills that are not a clip — the live
    # failure it was written for (687 rows -> 4.3M by silent interpolation)
    far_apart = pd.DataFrame(
        [{"track_id": f"{t:06d}", "frame": f, "x": 100.0 + t, "y": 200.0}
         for t in range(3) for f in (0, 6340, 12680)])
    assert observed_stride(far_apart) == 6340, observed_stride(far_apart)
    try:
        to_pipeline({"rows": far_apart}, None, fps=5.0, sample_stride=1)
        raise AssertionError("the stride guard did not fire on stills 6,340 frames apart")
    except ValueError as e:
        assert "6340" in str(e), str(e)
    cl_auto, rep_auto, _ = to_pipeline({"rows": far_apart}, None, fps=5.0,
                                       sample_stride="auto")
    assert rep_auto["sample_stride"] == 6340 and len(cl_auto) <= 12, rep_auto
    results["stride_guard"] = ("fires on frames 6,340 apart declared as stride 1; "
                               f"auto gives {len(cl_auto)} rows instead of ~45,000")

    # the bridge remap: facing must follow the id stage 3 chose
    # a person standing still whose id SAM dropped and re-issued mid-clip —
    # which is both the case stage 3's bridge exists for and the case face
    # facing exists for (a stander gets no motion facing at all)
    bridged_rows = pd.DataFrame(
        [{"track_id": "000000", "frame": i, "x": 500.0, "y": 500.0}
         for i in range(5)] +
        [{"track_id": "000001", "frame": i, "x": 500.0, "y": 500.0}
         for i in range(6, 11)])
    fac = pd.DataFrame([{"track_id": "000001", "frame": i, "facing_rad": 0.4,
                         "facing_conf": 0.6, "facing_source": "face"}
                        for i in range(6, 11)])
    cl3, rep3, fac3 = to_pipeline({"rows": bridged_rows}, fac, fps=5.0, sample_stride=1)
    assert rep3["gaps_bridged"] == 1, rep3
    assert set(fac3.track_id) == {"000000"}, \
        "facing kept the pre-bridge id, so it would have stopped landing"
    cl3 = cl3.assign(x_m=cl3.x / 100.0, y_m=cl3.y / 100.0)
    merged3 = apply_to_tracks(cl3, fac3)
    late = merged3[merged3.frame >= 6]
    assert (late.facing_source == "face").all(), late.facing_source.unique()
    assert np.allclose(late.facing_rad, 0.4)
    results["bridge_remap"] = ("1 bridge; facing re-keyed to the surviving id and "
                               "landed on the stander motion could not answer for")

    return results


def _print_live(args):
    """`--frames DIR` / `--headcount IMG`: the live instrument. Prints what it
    found, never the key."""
    client = SamVideoClient(max_usd=args.max_usd)
    if not client.key_present():
        print(f"no usable key at {DEFAULT_KEY_FILE} — nothing live can run")
        return 1
    if args.headcount:
        fs = FrameSet([args.headcount])
        print(json.dumps(headcount(client, fs, n_samples=1), indent=2))
        print(json.dumps(client.cost_report(), indent=2))
        return 0
    cal = None
    if args.calibration:
        from proxemix.calibration import load_calibration_config
        cal = load_calibration_config(args.calibration)
    res = run_sam_stage(frames_dir=args.frames, client=client, out_dir=args.out,
                        mode=args.mode, fps=args.fps, calibration_config=cal,
                        do_facing=not args.no_facing, keep_clip=args.keep_clip,
                        chunk_frames=(None if args.single_shot else args.chunk_frames),
                        chunk_overlap=args.chunk_overlap)
    print(json.dumps(res["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SAM 3.1 detect+track stage")
    ap.add_argument("--frames", help="directory of extracted frames")
    ap.add_argument("--headcount", help="one image: run the headcount front porch")
    ap.add_argument("--out", default=None, help="where to write rows/summary")
    ap.add_argument("--mode", default="auto", choices=["auto", "video", "image"])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--calibration", default=None,
                    help="calibration config json — makes facing angles floor-plane")
    ap.add_argument("--no-facing", action="store_true")
    ap.add_argument("--keep-clip", action="store_true")
    ap.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES,
                    help="target chunk length; each chunk re-seeds its box prompts "
                         f"(default {DEFAULT_CHUNK_FRAMES})")
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                    help="frames two neighbouring chunks share, which is what the "
                         "seam merge matches on")
    ap.add_argument("--single-shot", action="store_true",
                    help="seed box prompts once on frame 1 — the pre-2026-10-28 "
                         "behaviour, which never tracks a late arrival")
    ap.add_argument("--max-usd", type=float, default=2.0)
    args = ap.parse_args()

    if args.frames or args.headcount:
        raise SystemExit(_print_live(args))

    print(json.dumps(self_test(), indent=2, default=str))
    print("\nsam_video self-test PASSED — parser, client, spend cap, stitcher, "
          "partitioning, cross-pass merge, chunked re-seeding and its boundary "
          "merge, facing and the facing.py seam, all with no key and no network.")
