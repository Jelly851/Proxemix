"""Frame <-> time conversion shared by every export/report module. The
pipeline's own `t` column is always `frame / FPS` (proven by storage.py's
output: frame 120000 -> t 4800.0 on the 25fps Grand Central data), so this
is one constant rather than a lookup.
"""
FPS = 25.0


def frame_to_seconds(frame):
    return frame / FPS


def format_mmss(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def frame_citation(frame_start, frame_end):
    """Renders the spec's exact citation shape: [frame 4,210-4,540 / 02:48-03:02]."""
    t0 = frame_to_seconds(frame_start)
    t1 = frame_to_seconds(frame_end)
    return (
        f"[frame {int(frame_start):,}–{int(frame_end):,} / "
        f"{format_mmss(t0)}–{format_mmss(t1)}]"
    )
