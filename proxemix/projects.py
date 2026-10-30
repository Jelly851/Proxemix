"""The project store — a project is a directory, and it is also a bundle.

Jayden's call, 2026-10-30: *"we're missing any way to upload video files and make
projects."* Before this, the app could open exactly one pre-built demo. A project
fixes that without inventing a second data format: `data/projects/<slug>/` holds
the same `bundle_meta.json`, `proxemix.duckdb`, `fact_sheet.json`,
`report_statements.json` that `data/plaza_bundle/` holds, plus the source video,
`calibration.json`, `zones.json`, and a `project.json` carrying the stage
statuses. The plaza demo therefore opens as a project with no conversion.

Five stages, Resolve-style: Media, Calibrate, Analyze, Review, Report.
`stage_status()` is what greys out the bar, and it encodes two of Jayden's
rulings: **calibration is optional**, and **review is unreachable until an
analysis has actually finished**.

Uncalibrated projects, and why they are allowed
-----------------------------------------------
Mandatory calibration makes first use feel like homework, so skipping is
allowed. The cost is that distances come out in pixels. Rather than special-case
every downstream metre, a skipped calibration writes an *identity* homography —
image points at the frame corners, world points the same numbers — so the whole
pipeline runs unchanged and every figure is pixels wearing a metre's name. That
would be a lie if it were silent, so the record and the bundle both carry
`uncalibrated: true`, and the Report page prints a warning above the text.

    py -m proxemix.projects --selftest
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys
from datetime import datetime, timezone

PROJECT_FILE = "project.json"
CALIBRATION_FILE = "calibration.json"
ZONES_FILE = "zones.json"
DEFAULT_COST_CAP_USD = 5.00

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class ProjectError(RuntimeError):
    """Anything that would leave a project in a state the UI cannot read."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name):
    slug = _SLUG_STRIP.sub("-", str(name).strip().lower()).strip("-")
    return slug[:60] or "project"


def project_dir(projects_dir, slug):
    return pathlib.Path(projects_dir) / slug


def _record_path(projects_dir, slug):
    return project_dir(projects_dir, slug) / PROJECT_FILE


def create_project(projects_dir, name):
    """A new, empty project. The slug is made unique rather than overwriting:
    two projects called "Plaza" are two projects, not one lost one."""
    base = slugify(name)
    slug, n = base, 2
    while _record_path(projects_dir, slug).exists():
        slug = f"{base}-{n}"
        n += 1

    d = project_dir(projects_dir, slug)
    d.mkdir(parents=True, exist_ok=True)
    record = {
        "slug": slug,
        "name": str(name).strip(),
        "created_at": _now(),
        "updated_at": _now(),
        "video": None,
        "calibration": None,
        "zones": None,
        "analysis": None,
        "cost_cap_usd": DEFAULT_COST_CAP_USD,
    }
    return save_project(projects_dir, record)


def save_project(projects_dir, record):
    record["updated_at"] = _now()
    path = _record_path(projects_dir, record["slug"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def load_project(projects_dir, slug):
    path = _record_path(projects_dir, slug)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def list_projects(projects_dir):
    d = pathlib.Path(projects_dir)
    if not d.is_dir():
        return []
    out = []
    for child in d.iterdir():
        if not child.is_dir():
            continue
        record = load_project(projects_dir, child.name)
        if record:
            out.append(record)
    return sorted(out, key=lambda r: r.get("updated_at") or "", reverse=True)


def delete_project(projects_dir, slug):
    d = project_dir(projects_dir, slug)
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()


# --- media ------------------------------------------------------------------

def probe_video(path):
    """Duration, size and rate, read from the file rather than from its name."""
    import av

    path = pathlib.Path(path)
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        duration = None
        if container.duration:
            duration = float(container.duration) / 1_000_000.0
        elif stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        rate = None
        if stream.average_rate:
            rate = float(stream.average_rate)
        return {
            "duration_s": round(duration, 3) if duration else None,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "native_fps": round(rate, 3) if rate else None,
            "size_bytes": path.stat().st_size,
        }
    finally:
        container.close()


def attach_video(projects_dir, slug, src_path, filename=None):
    """Copy a clip into the project and record what it actually is.

    Probing happens *before* the record is written, so a file that is not a
    readable video fails here rather than three stages later inside SAM.
    """
    record = load_project(projects_dir, slug)
    if record is None:
        raise ProjectError(f"no project called {slug!r}")

    src = pathlib.Path(src_path)
    if not src.exists():
        raise ProjectError(f"{src} does not exist")

    name = pathlib.Path(filename or src.name).name
    if not name.lower().endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")):
        raise ProjectError(f"{name} is not a video file this can read")

    d = project_dir(projects_dir, slug)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / name

    # Replacing the footage invalidates everything derived from the old clip.
    for stale in ("poster.jpg",):
        try:
            (d / stale).unlink()
        except OSError:
            pass

    shutil.copyfile(src, dest)
    try:
        info = probe_video(dest)
    except Exception as e:
        try:
            dest.unlink()
        except OSError:
            pass
        raise ProjectError(f"{name} could not be read as video: {e}") from None

    record["video"] = {"filename": name, **info}
    record["analysis"] = None       # old results do not belong to a new clip
    return save_project(projects_dir, record)


# --- calibration ------------------------------------------------------------

def _write_calibration(projects_dir, slug, config):
    path = project_dir(projects_dir, slug) / CALIBRATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    return path


def set_calibration(projects_dir, slug, image_points, rect_size_m=None,
                    world_points_m=None, known_distance_m=None,
                    frame_width=None, frame_height=None):
    """Four clicked corners plus one real size. Validated by building it.

    `calibration.py` already accepts exactly this shape, so nothing new is
    computed here — but the homography is built once before saving, because a
    degenerate quad (three points in a line, two points identical) produces a
    config that only fails much later, inside the pipeline.
    """
    record = load_project(projects_dir, slug)
    if record is None:
        raise ProjectError(f"no project called {slug!r}")

    pts = [[float(p[0]), float(p[1])] for p in image_points]
    if len(pts) != 4:
        raise ProjectError("calibration needs exactly four corners")
    if len({(round(x, 3), round(y, 3)) for x, y in pts}) < 4:
        raise ProjectError("two of those corners are the same point")

    config = {"image_points": pts, "approximate": True,
              "note": f"Four corners clicked in the app, {_now()}."}
    if world_points_m:
        config["world_points_m"] = [[float(p[0]), float(p[1])] for p in world_points_m]
    elif rect_size_m:
        w, d = float(rect_size_m[0]), float(rect_size_m[1])
        if w <= 0 or d <= 0:
            raise ProjectError("the rectangle's size must be positive")
        config["rect_size_m"] = [w, d]
    elif known_distance_m:
        if float(known_distance_m) <= 0:
            raise ProjectError("the known distance must be positive")
        config["known_distance_m"] = float(known_distance_m)
    else:
        raise ProjectError("give the rectangle's real size in metres")

    if frame_width:
        config["frame_width"] = int(frame_width)
    if frame_height:
        config["frame_height"] = int(frame_height)

    from .calibration import build_homography
    try:
        build_homography(config)
    except Exception as e:
        raise ProjectError(
            f"those four points do not form a usable rectangle: {e}") from None

    _write_calibration(projects_dir, slug, config)
    record["calibration"] = config
    return save_project(projects_dir, record)


def identity_calibration(frame_width, frame_height):
    """The skipped-calibration config: pixels, honestly labelled as pixels."""
    w, h = float(frame_width), float(frame_height)
    return {
        "uncalibrated": True,
        "approximate": True,
        "note": ("NOT CALIBRATED. The image plane is mapped to itself, so every "
                 "distance below is in pixels wearing a metre's name. Figures are "
                 "comparable within this clip and meaningless between clips."),
        "image_points": [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
        "world_points_m": [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
    }


def ensure_calibration(projects_dir, slug):
    """The config the pipeline should use, calibrated or not.

    Returns `(config, path, uncalibrated)`. Called by the pipeline rather than
    by the UI, so a project Jayden chose to skip still runs end to end.
    """
    record = load_project(projects_dir, slug)
    if record is None:
        raise ProjectError(f"no project called {slug!r}")
    cfg = record.get("calibration")
    if cfg and not cfg.get("uncalibrated"):
        path = project_dir(projects_dir, slug) / CALIBRATION_FILE
        if not path.exists():
            _write_calibration(projects_dir, slug, cfg)
        return cfg, path, False

    video = record.get("video") or {}
    cfg = identity_calibration(video.get("width") or 1920,
                               video.get("height") or 1080)
    path = _write_calibration(projects_dir, slug, cfg)
    return cfg, path, True


# --- zones ------------------------------------------------------------------

def default_zones(calibration_config):
    """One zone covering the whole calibrated floor — Jayden's ruled default."""
    from .calibration import build_homography
    _, world = build_homography(calibration_config)
    x0, y0 = float(world[:, 0].min()), float(world[:, 1].min())
    x1, y1 = float(world[:, 0].max()), float(world[:, 1].max())
    pad_x, pad_y = (x1 - x0) * 0.05 or 1.0, (y1 - y0) * 0.05 or 1.0
    return {
        "note": "Default single zone covering the whole calibrated floor.",
        "frame": "floor_meters",
        "zones": [{
            "name": "whole_floor",
            "polygon_m": [[x0 - pad_x, y0 - pad_y], [x1 + pad_x, y0 - pad_y],
                          [x1 + pad_x, y1 + pad_y], [x0 - pad_x, y1 + pad_y]],
        }],
    }


def set_zones(projects_dir, slug, zones):
    record = load_project(projects_dir, slug)
    if record is None:
        raise ProjectError(f"no project called {slug!r}")
    cleaned = []
    for z in zones or []:
        name = str(z.get("name") or "").strip()
        poly = [[float(p[0]), float(p[1])] for p in (z.get("polygon_m") or [])]
        if not name:
            raise ProjectError("every zone needs a name")
        if len(poly) < 3:
            raise ProjectError(f"zone {name!r} needs at least three corners")
        cleaned.append({"name": name, "polygon_m": poly})
    if not cleaned:
        raise ProjectError("give at least one zone, or leave zones unset")

    config = {"note": f"Drawn in the app, {_now()}.",
              "frame": "floor_meters", "zones": cleaned}
    path = project_dir(projects_dir, slug) / ZONES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    record["zones"] = config
    return save_project(projects_dir, record)


def ensure_zones(projects_dir, slug, calibration_config):
    """The zones file the pipeline should use, defaulting to the whole floor."""
    record = load_project(projects_dir, slug) or {}
    path = project_dir(projects_dir, slug) / ZONES_FILE
    if record.get("zones") and path.exists():
        return path
    config = record.get("zones") or default_zones(calibration_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    return path


# --- stages -----------------------------------------------------------------

def stage_status(record):
    """What the stage bar shows: locked, ready, or done, for all five pages.

    Calibration is optional, so Analyze is reachable on footage alone. Review
    and Report stay locked until an analysis has genuinely finished — a project
    that failed mid-run must not offer a half-written bundle as though it were
    a result.
    """
    record = record or {}
    has_video = bool(record.get("video"))
    cal = record.get("calibration")
    analysis = record.get("analysis") or {}
    analyzed = analysis.get("state") == "done"

    return {
        "media": "done" if has_video else "ready",
        "calibrate": ("done" if (cal and not cal.get("uncalibrated"))
                      else "ready" if has_video else "locked"),
        "analyze": ("done" if analyzed
                    else "ready" if has_video else "locked"),
        "review": "done" if analyzed else "locked",
        "report": "done" if analyzed else "locked",
    }


# --- self-test --------------------------------------------------------------

def selftest():
    """No video, no network: the record lifecycle and the stage rules."""
    import tempfile

    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
        ok = ok and passed

    with tempfile.TemporaryDirectory() as tmp:
        p = create_project(tmp, "Eminonu Plaza")
        check("slug made from the name", p["slug"] == "eminonu-plaza", p["slug"])
        check("new project is empty", p["video"] is None and p["analysis"] is None)

        s = stage_status(p)
        check("media ready, rest locked",
              s["media"] == "ready" and s["calibrate"] == "locked"
              and s["analyze"] == "locked" and s["review"] == "locked")

        dup = create_project(tmp, "Eminonu Plaza")
        check("a duplicate name does not overwrite", dup["slug"] == "eminonu-plaza-2",
              dup["slug"])
        check("both are listed", len(list_projects(tmp)) == 2)

        # pretend footage, so the stage rules can be tested with no video file
        p["video"] = {"filename": "clip.mp4", "width": 1920, "height": 1080,
                      "duration_s": 20.0, "size_bytes": 1}
        save_project(tmp, p)
        s = stage_status(p)
        check("calibrate and analyze open with footage",
              s["calibrate"] == "ready" and s["analyze"] == "ready")
        check("review stays locked before analysis", s["review"] == "locked")

        p = set_calibration(tmp, p["slug"],
                            [[100, 500], [900, 500], [1000, 900], [50, 900]],
                            rect_size_m=[20.0, 12.0])
        check("calibration saves", p["calibration"]["rect_size_m"] == [20.0, 12.0])
        check("calibrate reads done", stage_status(p)["calibrate"] == "done")
        check("calibration file written",
              (project_dir(tmp, p["slug"]) / CALIBRATION_FILE).exists())

        try:
            set_calibration(tmp, p["slug"], [[1, 1], [1, 1], [2, 2], [3, 3]],
                            rect_size_m=[10, 10])
            check("a degenerate quad is refused", False, "it was accepted")
        except ProjectError:
            check("a degenerate quad is refused", True)

        cfg, path, uncal = ensure_calibration(tmp, p["slug"])
        check("calibrated project uses its own config", uncal is False)

        p2 = dup
        p2["video"] = {"filename": "c.mp4", "width": 640, "height": 480,
                       "duration_s": 5.0, "size_bytes": 1}
        save_project(tmp, p2)
        cfg2, path2, uncal2 = ensure_calibration(tmp, p2["slug"])
        check("skipping calibration still yields a usable config", uncal2 is True)
        check("the fallback says so in its own note",
              "NOT CALIBRATED" in cfg2["note"])
        zpath = ensure_zones(tmp, p2["slug"], cfg2)
        zones = json.loads(zpath.read_text(encoding="utf-8"))
        check("zones default to one whole-floor zone", len(zones["zones"]) == 1,
              zones["zones"][0]["name"])

        p2["analysis"] = {"state": "done"}
        check("review opens once analysis is done",
              stage_status(p2)["review"] == "done")
        p2["analysis"] = {"state": "failed", "error": "SAM returned no tracks"}
        check("a failed analysis does not open review",
              stage_status(p2)["review"] == "locked")

        check("delete removes the directory", delete_project(tmp, p2["slug"]))
        check("and it is gone from the list", len(list_projects(tmp)) == 1)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else selftest())
