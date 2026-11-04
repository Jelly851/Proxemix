"""The Analyze stage — one video in, a finished bundle out.

This is `scratch/build_plaza_bundle.py` generalised. That script proved the
chain end to end on real footage; everything here is the same sequence with the
plaza's hard-coded paths replaced by a project, plus three things a one-off
script never needed: progress a UI can poll, a cost ceiling checked before
spending, and a failure that names its own step.

The chain, and the module that already does each part:

    frames    ingest.extract_frames
    detect    sam_video.run_sam_stage          (the only step that costs money)
    clean     track_cleaning.clean_tracks
    store     storage.store_cleaned_tracks
    calibrate calibration.apply_calibration_to_tracks
    metrics   metrics_pipeline.run_metrics
    facts     fact_sheet.write_fact_sheet
    report    report_writer + report_verifier
    video     ffmpeg transcode + bundle_meta.json

Two refusals are deliberate
---------------------------
**No step swallows an exception and continues.** A pipeline that "succeeds" with
zero tracks is the worst outcome available here, because the report still
renders and still looks plausible — it would simply be about nobody. The detect
step fails loudly on an empty result rather than letting eight later steps
produce confident emptiness.

**Cost is refused before it is spent.** `estimate_cost_usd` is shown to Jayden
before he commits, and `max_usd` reaches `SamVideoClient`, whose cap refuses the
call that would cross it *before* sending.

    py -m proxemix.pipeline --selftest
    py -m proxemix.pipeline --projects data/projects --slug my-clip
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone

from . import projects
from .sam_video import estimate_pass_cost

PIPELINE_STEPS = (
    ("frames", "Extract frames"),
    ("detect", "Detect and track people"),
    ("clean", "Clean and bridge tracks"),
    ("store", "Store tracks"),
    ("calibrate", "Project to the floor plane"),
    ("metrics", "Compute events and metrics"),
    ("facts", "Extract citable facts"),
    ("report", "Draft and verify the report"),
    ("video", "Prepare the playable video"),
)

# A run whose tracks collapse like this is almost certainly a moving camera or
# footage too dark to read. The pipeline still finishes — but it says so, rather
# than presenting nonsense confidently.
IMPLAUSIBLE_MIN_TRACKS = 2

_JOBS = {}
_JOBS_LOCK = threading.Lock()


class PipelineError(RuntimeError):
    """A step failed. Carries the step key so the UI can point at it."""

    def __init__(self, step, message):
        super().__init__(message)
        self.step = step


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_cost_usd(n_frames, n_concepts=2):
    """What the detect step would cost. The only step that spends anything."""
    return round(estimate_pass_cost(int(n_frames), n_concepts, "video"), 4)


def probe_video(path):
    return projects.probe_video(path)


def run_pipeline(projects_dir, slug, fps=5.0, max_usd=None, concepts=None,
                 on_progress=None, client=None, should_cancel=None):
    """The whole chain for one project. Returns a summary dict.

    `on_progress(step_key, state, detail)` fires at every step boundary, where
    state is "running", "done" or "failed".
    """
    from . import storage, track_cleaning
    from .calibration import apply_calibration_to_tracks, load_calibration_config
    from .fact_sheet import write_fact_sheet
    from .ingest import extract_frames
    from .metrics_pipeline import run_metrics
    from .report_verifier import verify_and_render
    from .report_writer import MuseWonderReportWriter, StubReportWriter
    from .sam_video import SamVideoClient, run_sam_stage

    record = projects.load_project(projects_dir, slug)
    if record is None:
        raise PipelineError("frames", f"no project called {slug!r}")
    if not record.get("video"):
        raise PipelineError("frames", "this project has no video yet")

    d = projects.project_dir(projects_dir, slug)
    src_video = d / record["video"]["filename"]
    fps = float(fps)
    cap = float(max_usd if max_usd is not None
                else record.get("cost_cap_usd") or projects.DEFAULT_COST_CAP_USD)

    def emit(step, state, detail=""):
        if on_progress:
            on_progress(step, state, detail)

    def cancelled():
        return bool(should_cancel and should_cancel())

    summary = {"slug": slug, "fps": fps, "warnings": [], "cost_usd": 0.0}

    # --- frames ------------------------------------------------------------
    emit("frames", "running")
    frames_dir = d / "frames"
    try:
        extract_frames(src_video, frames_dir, fps=fps)
        n_frames = len(list(frames_dir.glob("*.jpg")))
    except Exception as e:
        emit("frames", "failed", str(e)[:200])
        raise PipelineError("frames", f"could not read the video: {e}") from None
    if n_frames < 2:
        emit("frames", "failed", f"{n_frames} frame(s)")
        raise PipelineError("frames", f"only {n_frames} frame(s) came out of the clip")
    summary["n_frames"] = n_frames
    emit("frames", "done", f"{n_frames} frames at {fps} fps")
    if cancelled():
        raise PipelineError("frames", "cancelled")

    # --- calibration config, before detect, because SAM's facing wants it ---
    cal_cfg, cal_path, uncalibrated = projects.ensure_calibration(projects_dir, slug)
    zones_path = projects.ensure_zones(projects_dir, slug, cal_cfg)
    summary["uncalibrated"] = uncalibrated
    if uncalibrated:
        summary["warnings"].append(
            "No calibration: distances are pixels, comparable within this clip only.")

    # --- detect ------------------------------------------------------------
    emit("detect", "running", f"estimated ${estimate_cost_usd(n_frames):.2f}")
    if client is None:
        client = SamVideoClient(max_usd=cap)
    try:
        stage = run_sam_stage(frames_dir=str(frames_dir), fps=fps,
                              calibration_config=cal_cfg, client=client)
    except Exception as e:
        emit("detect", "failed", str(e)[:200])
        raise PipelineError("detect", f"tracking failed: {e}") from None
    raw = stage["rows"]
    summary["cost_usd"] = round(float(getattr(client, "spent_usd", 0.0)), 4)
    if raw is None or not len(raw):
        emit("detect", "failed", "no people found")
        raise PipelineError(
            "detect",
            "Nothing was tracked in this clip. That usually means the camera is "
            "moving, the footage is too dark, or there are no people in frame — "
            "this pipeline assumes a fixed camera looking at a floor.")
    n_raw_tracks = int(raw["track_id"].nunique())
    emit("detect", "done",
         f"{n_raw_tracks} tracks, ${summary['cost_usd']:.3f}")
    if cancelled():
        raise PipelineError("detect", "cancelled")

    # SAM numbers its frames from 1; every downstream frame-to-time mapping
    # assumes 0-based, so a missed shift puts every event one frame early.
    raw = raw.copy()
    raw["frame"] = raw["frame"] - raw["frame"].min()

    # --- clean -------------------------------------------------------------
    emit("clean", "running")
    try:
        cleaned, clean_report = track_cleaning.clean_tracks(raw, fps=fps,
                                                            sample_stride=1)
    except Exception as e:
        emit("clean", "failed", str(e)[:200])
        raise PipelineError("clean", f"cleaning failed: {e}") from None
    if not len(cleaned):
        emit("clean", "failed", "every track was discarded")
        raise PipelineError("clean", "every track was discarded during cleaning")
    summary["clean_report"] = clean_report
    emit("clean", "done", f"{clean_report.get('tracks_after')} tracks kept")

    n_tracks = int(cleaned["track_id"].nunique())
    if n_tracks < IMPLAUSIBLE_MIN_TRACKS:
        summary["warnings"].append(
            f"Only {n_tracks} track(s) survived cleaning. Read this result as "
            "unreliable — a fixed camera on a populated floor gives far more.")

    # --- store -------------------------------------------------------------
    emit("store", "running")
    db_path = d / "proxemix.duckdb"
    try:
        if db_path.exists():
            db_path.unlink()
        storage.store_cleaned_tracks(str(db_path), cleaned)
    except Exception as e:
        emit("store", "failed", str(e)[:200])
        raise PipelineError("store", f"could not store tracks: {e}") from None
    emit("store", "done", f"{len(cleaned):,} rows")

    # --- calibrate ---------------------------------------------------------
    emit("calibrate", "running")
    try:
        cfg = load_calibration_config(str(cal_path))
        calibrated = apply_calibration_to_tracks(cleaned, cfg)
    except Exception as e:
        emit("calibrate", "failed", str(e)[:200])
        raise PipelineError("calibrate", f"calibration failed: {e}") from None
    emit("calibrate", "done", "pixels" if uncalibrated else "metres")

    # --- metrics -----------------------------------------------------------
    emit("metrics", "running")
    con = storage.init_db(str(db_path))
    try:
        event_counts = run_metrics(
            con, calibrated, cfg, str(zones_path),
            contact_threshold_m=1.5, contact_min_duration_s=2.0,
            pairwise_max_radius_m=3.6, od_grid_cells=2, outlier_top_n=10)
    except Exception as e:
        con.close()
        emit("metrics", "failed", str(e)[:200])
        raise PipelineError("metrics", f"metrics failed: {e}") from None
    summary["event_counts"] = event_counts
    emit("metrics", "done",
         f"{sum(int(v) for v in event_counts.values()):,} events")

    # --- facts -------------------------------------------------------------
    emit("facts", "running")
    try:
        facts = write_fact_sheet(con, str(d / "fact_sheet.json"))
    except Exception as e:
        con.close()
        emit("facts", "failed", str(e)[:200])
        raise PipelineError("facts", f"fact extraction failed: {e}") from None
    emit("facts", "done", f"{len(facts)} facts")

    # --- report ------------------------------------------------------------
    emit("report", "running")
    try:
        # Muse Wonder drafts the findings when a key is on disk, and the
        # deterministic stub does when it is not — so this stage never fails
        # for want of a model, and the verifier below judges either draft the
        # same way.
        writer = StubReportWriter()
        try:
            from .muse import MuseClient
            client = MuseClient()
            if client.key_present():
                writer = MuseWonderReportWriter(client=client)
        except Exception:
            pass
        draft = writer.draft(facts)
        statements, dropped, markdown = verify_and_render(
            draft, facts, title=f"Proxemix — {record['name']}")
        (d / "report.md").write_text(markdown, encoding="utf-8")
        (d / "report_statements.json").write_text(
            json.dumps(statements, indent=2), encoding="utf-8")
    except Exception as e:
        con.close()
        emit("report", "failed", str(e)[:200])
        raise PipelineError("report", f"report failed: {e}") from None
    finally:
        con.close()
    summary["n_facts"] = len(facts)
    summary["n_dropped_poison_check"] = len(dropped)
    emit("report", "done", f"{len(statements)} statements, {len(dropped)} dropped")

    # --- video + bundle meta ------------------------------------------------
    emit("video", "running")
    out_video = d / "video.mp4"
    try:
        _transcode(src_video, out_video)
        duration = _duration(out_video) or record["video"].get("duration_s")
    except Exception as e:
        emit("video", "failed", str(e)[:200])
        raise PipelineError("video", f"could not prepare the video: {e}") from None

    meta = {
        "fps": fps,
        "width": record["video"]["width"],
        "height": record["video"]["height"],
        "duration_s": duration,
        "video_file": "video.mp4",
        "db_file": "proxemix.duckdb",
        "source": f"{record['video']['filename']} — analyzed {_now()}",
        "uncalibrated": uncalibrated,
        "clean_report": clean_report,
        "event_counts": event_counts,
        "n_facts": len(facts),
        "n_dropped_poison_check": len(dropped),
        "warnings": summary["warnings"],
    }
    (d / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    emit("video", "done", f"{duration:.1f}s" if duration else "ready")

    summary["counts"] = event_counts
    return summary


def _transcode(src, dest):
    """Re-encode for the browser: dense keyframes so seeking lands exactly."""
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-g", "12", "-bf", "0", "-movflags", "+faststart", "-an", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + (r.stderr or "")[-800:])
    return dest


def _duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


# --- jobs -------------------------------------------------------------------

def start_job(projects_dir, slug, **kwargs):
    """Run the pipeline on a background thread and return a job id.

    One project runs at a time: a second request while a job is live returns
    that job rather than starting a competing writer on the same directory.
    """
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job["slug"] == slug and job["state"] == "running":
                return job["job_id"]

        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "slug": slug,
            "state": "running",
            "steps": [{"key": k, "label": label, "state": "waiting", "detail": ""}
                      for k, label in PIPELINE_STEPS],
            "cost_usd": 0.0,
            "error": None,
            "warnings": [],
            "started_at": _now(),
            "finished_at": None,
            "cancel": False,
        }
        _JOBS[job_id] = job

    def on_progress(step_key, state, detail=""):
        with _JOBS_LOCK:
            for s in job["steps"]:
                if s["key"] == step_key:
                    s["state"] = state
                    s["detail"] = detail or s["detail"]

    def should_cancel():
        with _JOBS_LOCK:
            return job["cancel"]

    def work():
        record = projects.load_project(projects_dir, slug) or {}
        record["analysis"] = {"state": "running", "job_id": job_id,
                              "started_at": job["started_at"]}
        projects.save_project(projects_dir, record)
        try:
            summary = run_pipeline(projects_dir, slug, on_progress=on_progress,
                                   should_cancel=should_cancel, **kwargs)
            with _JOBS_LOCK:
                job["state"] = "done"
                job["cost_usd"] = summary.get("cost_usd", 0.0)
                job["warnings"] = summary.get("warnings", [])
                job["finished_at"] = _now()
            record = projects.load_project(projects_dir, slug) or {}
            record["analysis"] = {
                "state": "done", "job_id": job_id,
                "started_at": job["started_at"], "finished_at": job["finished_at"],
                "cost_usd": summary.get("cost_usd", 0.0),
                "counts": summary.get("counts", {}),
                "warnings": summary.get("warnings", []),
                "uncalibrated": summary.get("uncalibrated", False),
            }
            projects.save_project(projects_dir, record)
        except BaseException as e:
            step = getattr(e, "step", None)
            with _JOBS_LOCK:
                job["state"] = "failed"
                job["error"] = f"{step + ': ' if step else ''}{e}"
                job["finished_at"] = _now()
                job["traceback"] = traceback.format_exc()[-1200:]
                for s in job["steps"]:
                    if s["state"] == "running":
                        s["state"] = "failed"
            record = projects.load_project(projects_dir, slug) or {}
            record["analysis"] = {"state": "failed", "job_id": job_id,
                                  "started_at": job["started_at"],
                                  "finished_at": job["finished_at"],
                                  "error": str(e)[:500], "failed_step": step}
            projects.save_project(projects_dir, record)

    threading.Thread(target=work, name=f"proxemix-{slug}", daemon=True).start()
    return job_id


def job_status(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in job.items() if k != "cancel"}


def list_jobs():
    with _JOBS_LOCK:
        return [dict(j) for j in _JOBS.values()]


def cancel_job(job_id):
    """Asks the run to stop at its next step boundary. A step already inside
    SAM finishes — there is no way to un-send a request that is already paid for."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["state"] != "running":
            return False
        job["cancel"] = True
        return True


# --- self-test --------------------------------------------------------------

def selftest():
    """The job machinery and the refusals, with no network and no video."""
    import tempfile
    import time

    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
        ok = ok and passed

    check("nine steps, in pipeline order", len(PIPELINE_STEPS) == 9
          and PIPELINE_STEPS[0][0] == "frames" and PIPELINE_STEPS[-1][0] == "video")
    check("cost rises with frames",
          estimate_cost_usd(1000) > estimate_cost_usd(100))

    with tempfile.TemporaryDirectory() as tmp:
        p = projects.create_project(tmp, "No Footage")

        # a project with no video must fail at the first step, not later
        try:
            run_pipeline(tmp, p["slug"])
            check("a project with no video is refused", False, "it ran")
        except PipelineError as e:
            check("a project with no video is refused", e.step == "frames", e.step)

        # the job wrapper must record that failure on the project record
        job_id = start_job(tmp, p["slug"])
        for _ in range(100):
            s = job_status(job_id)
            if s["state"] != "running":
                break
            time.sleep(0.05)
        s = job_status(job_id)
        check("the job reports failure", s["state"] == "failed", s.get("error", ""))
        check("the error names its step", "frames" in (s.get("error") or ""))
        record = projects.load_project(tmp, p["slug"])
        check("the project records the failure",
              record["analysis"]["state"] == "failed")
        check("a failed analysis leaves review locked",
              projects.stage_status(record)["review"] == "locked")
        check("job status hides its cancel flag", "cancel" not in s)
        check("cancelling a finished job returns false", cancel_job(job_id) is False)

    return 0 if ok else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run the Proxemix analysis pipeline.")
    ap.add_argument("--projects", default="data/projects")
    ap.add_argument("--slug")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.slug:
        ap.error("give --slug or --selftest")

    def show(step, state, detail=""):
        print(f"  {state:<8} {step:<10} {detail}")

    try:
        summary = run_pipeline(args.projects, args.slug, fps=args.fps,
                               max_usd=args.max_usd, on_progress=show)
    except PipelineError as e:
        print(f"\nFAILED at {e.step}: {e}", file=sys.stderr)
        return 1
    print("\n" + json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
