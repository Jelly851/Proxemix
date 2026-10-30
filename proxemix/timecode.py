"""Frame <-> time conversion shared by every export/report module.

The pipeline's own `t` column is always `frame / fps`, proven by storage.py's
output: frame 120000 -> t 4800.0 on the 25 fps Grand Central data.

`FPS = 25.0` is the **default**, not the truth. It is Grand Central's rate, and
every number banked against that dataset was rendered with it, so it stays the
default. But a bundle carries its own rate — the plaza demo runs at 5 fps — and
a citation rendered at 25 fps on a 5 fps clip names a moment five times too
early: frame 89 printed as 00:03 when it is really 00:17. The player seeks
correctly (the UI reads `bundle.fps`), so the frame jump lands and only the
printed timestamp lies, which is the worst version of the fault — it looks right.
So every function here takes an optional `fps`, and anything rendering a citation
for a bundle passes that bundle's own rate.
"""
FPS = 25.0


def _fps(fps=None):
    if fps is None:
        return FPS
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return fps


def frame_to_seconds(frame, fps=None):
    return frame / _fps(fps)


def format_mmss(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def frame_citation(frame_start, frame_end, fps=None):
    """Renders the spec's exact citation shape: [frame 4,210-4,540 / 02:48-03:02]."""
    rate = _fps(fps)
    t0 = frame_to_seconds(frame_start, rate)
    t1 = frame_to_seconds(frame_end, rate)
    return (
        f"[frame {int(frame_start):,}–{int(frame_end):,} / "
        f"{format_mmss(t0)}–{format_mmss(t1)}]"
    )
