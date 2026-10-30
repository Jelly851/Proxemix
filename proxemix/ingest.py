"""Stage 1 — ingest: re-encode a video for dense keyframes, then extract
frames at a fixed fps via PyAV (frame-exact; OpenCV's seek is not).
"""
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import av

EXTRACT_FPS = 5


def reencode(src_path, out_path, gop=12, fps=None):
    """ffmpeg re-encode: dense keyframes, no B-frames, moov atom up front."""
    src_path = str(src_path)
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-g", str(gop), "-bf", "0", "-movflags", "+faststart",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += [out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed: {result.stderr[-2000:]}")
    return out_path


def extract_frames(video_path, out_dir, fps=EXTRACT_FPS):
    """Extract frames at `fps` using PyAV native timestamps (not seek-based).

    Walks every decoded frame and keeps the ones whose presentation time
    crosses the next sample boundary, so sampling stays frame-exact
    regardless of the source's actual encoded frame rate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # `Fraction(1, fps)` refuses a float, so a bundle declaring fps 5.0 rather
    # than 5 crashed here. Going through Fraction(fps) accepts both and keeps
    # the exact rational arithmetic the frame-exact sampling depends on.
    sample_period = 1 / Fraction(fps).limit_denominator(10000)
    next_sample_time = Fraction(0)
    manifest = []
    frame_idx = 0

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        timestamp = Fraction(frame.pts) * frame.time_base
        if timestamp < next_sample_time:
            continue
        img = frame.to_image()
        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = out_dir / frame_name
        img.save(frame_path, quality=95)
        manifest.append({
            "index": frame_idx,
            "timestamp": float(timestamp),
            "path": str(frame_path),
        })
        frame_idx += 1
        next_sample_time += sample_period

    container.close()

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": len(manifest),
            "frames": manifest,
        }, f, indent=2)

    return manifest_path


def run_ingest(video_path, work_dir, fps=EXTRACT_FPS):
    """Full stage-1 pipeline: re-encode then extract frames."""
    work_dir = Path(work_dir)
    reencoded_path = work_dir / "reencoded.mp4"
    frames_dir = work_dir / "frames"
    reencode(video_path, reencoded_path)
    manifest_path = extract_frames(reencoded_path, frames_dir, fps=fps)
    return manifest_path
