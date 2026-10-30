"""Stage 10 — the app backend. Projects, the five-stage pipeline, and the demo UI.

What changed on 2026-10-30
--------------------------
This used to serve exactly one pre-built bundle directory and nothing else: you
could look at the plaza demo and that was the product. Jayden's call — *"we're
missing any way to upload video files and make projects"* — turned it into a
tool. A **project** is a directory under `data/projects/`, and it *is* a bundle:
same `bundle_meta.json`, `proxemix.duckdb`, `fact_sheet.json` layout, plus the
source video, the calibration, and a `project.json` holding stage statuses. The
plaza demo opens as a project with no conversion.

Five stages, in order, modelled on DaVinci Resolve's page bar: Media, Calibrate,
Analyze, Review, Report. `projects.stage_status()` decides which are reachable,
so a clip that was never analyzed cannot be reviewed.

Backwards compatibility is deliberate
-------------------------------------
`--bundle DIR` still works and still serves that directory. Every bundle-reading
route takes an optional `?project=` and falls back to the launch bundle, so the
existing browser proofs (`scratch/ui_test_plaza.py`, `scratch/ui_test_search.py`)
keep passing unchanged.

Run: py -m proxemix.serve                      # projects UI
     py -m proxemix.serve --bundle data/plaza_bundle   # straight to one bundle
"""
import argparse
import io
import json
import os
import shutil
import tempfile

import duckdb
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional

from pydantic import BaseModel

from . import pipeline, projects
from .chat_stub import StubChatBackend
from .demo_bundle import build_demo_bundle
from .promptable import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_STRIDE,
    SearchError,
    list_searches,
    load_cached,
)
from .promptable import search as concept_search
from .timecode import frame_citation

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
DEFAULT_PROJECTS_DIR = os.path.join("data", "projects")

# What one search request from the browser may spend, and what the whole server
# may spend before it refuses. A text box wired to a paid API with no ceiling is
# the one way this UI could cost real money by accident.
SEARCH_MAX_USD_PER_REQUEST = 0.25
SEARCH_MAX_USD_PER_SESSION = 2.00

EVENT_LABELS = {
    "contact": lambda r: f"Contact: {r['actor_ids']} ({r['value']:.1f}s)",
    "hall_band": lambda r: f"Hall band '{r['zone']}': {r['actor_ids']} ({r['value']:.1f}s)",
    "hall_transition": lambda r: f"Hall transition {r['zone']}: {r['actor_ids']}",
    "zone_dwell": lambda r: f"Zone dwell '{r['zone']}': {r['actor_ids']} ({r['value']:.1f}s)",
    "zone_first_entry": lambda r: f"First entry '{r['zone']}': {r['actor_ids']} (+{r['value']:.1f}s)",
    "group": lambda r: f"Group ({int(r['value'])} members): {r['actor_ids']}",
}


def _event_label(row):
    fn = EVENT_LABELS.get(row["type"])
    if fn:
        try:
            return fn(row)
        except Exception:
            pass
    return f"{row['type']}: {row['actor_ids']}"


# Every optional field is spelled Optional[...], not `x: str = None`. Pydantic
# reads the bare form as "a string, defaulting to None" and then rejects an
# explicit `null` in the body with a 422 — which is exactly how the Search tab
# came back empty: the browser sends `project: null` when no project is open.
class ChatRequest(BaseModel):
    question: str
    project: Optional[str] = None


class SearchRequest(BaseModel):
    concept: str
    stride: int = DEFAULT_STRIDE
    refresh: bool = False
    project: Optional[str] = None


class NewProject(BaseModel):
    name: str


class CalibrationRequest(BaseModel):
    image_points: List[list]
    rect_size_m: Optional[List[float]] = None
    known_distance_m: Optional[float] = None
    world_points_m: Optional[List[list]] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None


class ZonesRequest(BaseModel):
    zones: List[dict]


class AnalyzeRequest(BaseModel):
    fps: float = 5.0
    max_usd: Optional[float] = None


class BundleView:
    """One bundle directory's readable contents, loaded once and reused.

    A project's files change only when its pipeline runs, so caching is safe;
    `invalidate()` is called the moment an analysis finishes.
    """

    def __init__(self, bundle_dir):
        self.dir = str(bundle_dir)
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        with open(os.path.join(self.dir, "bundle_meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(self.dir, "fact_sheet.json"), encoding="utf-8") as f:
            self.fact_sheet = json.load(f)
        with open(os.path.join(self.dir, "report_statements.json"), encoding="utf-8") as f:
            self.statements = json.load(f)
        self.facts_by_id = {f["fact_id"]: f for f in self.fact_sheet}
        self.db_path = os.path.join(self.dir, self.meta["db_file"])
        # The bundle's own frame rate, not a module default. A citation rendered
        # at 25 fps on a 5 fps clip names a moment five times too early, and the
        # player still seeks correctly — so only the printed timestamp lies.
        self.fps = float(self.meta.get("fps") or 0) or None
        self._loaded = True

    def ready(self):
        return os.path.exists(os.path.join(self.dir, "bundle_meta.json"))

    def invalidate(self):
        self._loaded = False

    def cite(self, frame_start, frame_end):
        self._load()
        return frame_citation(frame_start, frame_end, fps=self.fps)

    def fact_citations(self, fact_ids):
        self._load()
        return [
            {"fact_id": cid,
             "frame_start": self.facts_by_id[cid]["frame_start"],
             "frame_end": self.facts_by_id[cid]["frame_end"],
             "citation": self.cite(self.facts_by_id[cid]["frame_start"],
                                   self.facts_by_id[cid]["frame_end"])}
            for cid in fact_ids if cid in self.facts_by_id
        ]


def extract_poster(video_path, out_path):
    """One still from the clip, for the calibration canvas and the project card.

    Written here rather than in the pipeline because it is a view concern: the
    picture exists so Jayden can click four corners on it, and it is wanted
    before any analysis has run.
    """
    if os.path.exists(out_path):
        return out_path
    import av
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame.to_image().save(out_path, quality=92)
            return out_path
    finally:
        container.close()
    return None


def create_app(bundle_dir=None, projects_dir=DEFAULT_PROJECTS_DIR,
               auto_open_bundle=False):
    # The legacy single-bundle mode. Given a directory that is not yet built,
    # the synthetic fixture is built into it, exactly as before.
    launch_view = None
    if bundle_dir:
        if not os.path.isdir(bundle_dir) or not os.path.exists(
                os.path.join(bundle_dir, "bundle_meta.json")):
            build_demo_bundle(bundle_dir)
        launch_view = BundleView(bundle_dir)

    os.makedirs(projects_dir, exist_ok=True)
    views = {}
    chat_backend = StubChatBackend()
    spent = {"usd": 0.0}

    def view_for(slug=None):
        """The bundle a request is about: a named project, or the launch bundle."""
        if slug:
            d = str(projects.project_dir(projects_dir, slug))
            if d not in views:
                views[d] = BundleView(d)
            return views[d]
        if launch_view is None:
            raise FileNotFoundError("no project named and no launch bundle")
        return launch_view

    app = FastAPI(title="Proxemix")
    app.mount("/static", StaticFiles(directory=os.path.join(WEBUI_DIR, "static")),
              name="static")
    if bundle_dir:
        app.mount("/media", StaticFiles(directory=bundle_dir), name="media")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEBUI_DIR, "index.html"))

    # ---- projects ---------------------------------------------------------

    def _decorated(record):
        return {**record, "stages": projects.stage_status(record)}

    @app.get("/api/projects")
    def api_projects():
        """Jayden's own projects, plus the launch bundle offered as a demo card.

        `--bundle` used to make the app skip the project manager entirely, which
        meant the launcher opened the plaza demo and he never saw any of the new
        pages. The bundle now appears as one read-only entry in the list, and
        only `--demo` reinstates the old skip-straight-there behaviour, which is
        what the two browser proofs run against.
        """
        rows = [_decorated(r) for r in projects.list_projects(projects_dir)]
        demo = None
        if launch_view is not None and launch_view.ready():
            demo = {"slug": None, "name": "Plaza demo", "demo": True,
                    "video": True, "analysis": {"state": "done"},
                    "stages": {"media": "locked", "calibrate": "locked",
                               "analyze": "locked", "review": "done",
                               "report": "done"}}
        return {"projects": rows, "demo": demo,
                "has_launch_bundle": launch_view is not None,
                "auto_open": bool(auto_open_bundle and launch_view is not None)}

    @app.post("/api/projects")
    def api_create_project(req: NewProject):
        name = (req.name or "").strip()
        if not name:
            return JSONResponse({"error": "A project needs a name."}, status_code=400)
        return _decorated(projects.create_project(projects_dir, name))

    @app.get("/api/projects/{slug}")
    def api_project(slug: str):
        record = projects.load_project(projects_dir, slug)
        if record is None:
            return JSONResponse({"error": f"No project called {slug!r}."}, status_code=404)
        return _decorated(record)

    @app.delete("/api/projects/{slug}")
    def api_delete_project(slug: str):
        return {"deleted": projects.delete_project(projects_dir, slug)}

    @app.post("/api/projects/{slug}/video")
    async def api_upload_video(slug: str, request: Request):
        """The drop target.

        The clip arrives as the raw request body with its name in a header,
        rather than as a multipart form. Multipart would pull in
        `python-multipart`, and this repo deliberately carries no dependency it
        does not need — a video upload is one file, so the envelope buys nothing.

        Streamed to a temp file first, so a failed upload never leaves a
        half-written video sitting in the project as though it were whole.
        """
        if projects.load_project(projects_dir, slug) is None:
            return JSONResponse({"error": f"No project called {slug!r}."}, status_code=404)
        filename = os.path.basename(request.headers.get("x-filename") or "upload.mp4")
        suffix = os.path.splitext(filename)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            async for chunk in request.stream():
                tmp.write(chunk)
            tmp.close()
            record = projects.attach_video(projects_dir, slug, tmp.name,
                                           filename=filename)
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:300]}"},
                                status_code=400)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        return _decorated(record)

    @app.get("/api/projects/{slug}/poster")
    def api_poster(slug: str):
        record = projects.load_project(projects_dir, slug)
        if record is None or not record.get("video"):
            return JSONResponse({"error": "No video on this project yet."},
                                status_code=404)
        d = projects.project_dir(projects_dir, slug)
        src = os.path.join(str(d), record["video"]["filename"])
        out = os.path.join(str(d), "poster.jpg")
        try:
            extract_poster(src, out)
        except Exception as e:
            return JSONResponse({"error": f"Could not read a frame: {e}"},
                                status_code=400)
        return FileResponse(out)

    @app.get("/api/projects/{slug}/source")
    def api_source_video(slug: str):
        """The uploaded clip itself — what the calibration page plays."""
        record = projects.load_project(projects_dir, slug)
        if record is None or not record.get("video"):
            return JSONResponse({"error": "No video on this project yet."},
                                status_code=404)
        d = projects.project_dir(projects_dir, slug)
        return FileResponse(os.path.join(str(d), record["video"]["filename"]))

    @app.post("/api/projects/{slug}/calibration")
    def api_calibration(slug: str, req: CalibrationRequest):
        if len(req.image_points or []) != 4:
            return JSONResponse(
                {"error": "Calibration needs exactly four corners, clicked "
                          "clockwise from the top-left."}, status_code=400)
        if not (req.rect_size_m or req.known_distance_m or req.world_points_m):
            return JSONResponse(
                {"error": "Give the rectangle's real size in metres."},
                status_code=400)
        try:
            record = projects.set_calibration(
                projects_dir, slug, req.image_points,
                rect_size_m=req.rect_size_m,
                known_distance_m=req.known_distance_m,
                world_points_m=req.world_points_m,
                frame_width=req.frame_width, frame_height=req.frame_height)
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:300]}"},
                                status_code=400)
        return _decorated(record)

    @app.post("/api/projects/{slug}/zones")
    def api_zones(slug: str, req: ZonesRequest):
        try:
            record = projects.set_zones(projects_dir, slug, req.zones)
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:300]}"},
                                status_code=400)
        return _decorated(record)

    # ---- the pipeline -----------------------------------------------------

    @app.get("/api/projects/{slug}/estimate")
    def api_estimate(slug: str, fps: float = 5.0):
        """What an analysis would cost, before anything is committed."""
        record = projects.load_project(projects_dir, slug)
        if record is None or not record.get("video"):
            return JSONResponse({"error": "Add a video first."}, status_code=400)
        duration = float(record["video"].get("duration_s") or 0)
        n_frames = max(1, int(round(duration * float(fps))))
        return {"n_frames": n_frames, "fps": fps,
                "usd": round(pipeline.estimate_cost_usd(n_frames), 4),
                "cost_cap_usd": record.get("cost_cap_usd"),
                "steps": [{"key": k, "label": label}
                          for k, label in pipeline.PIPELINE_STEPS]}

    @app.post("/api/projects/{slug}/analyze")
    def api_analyze(slug: str, req: AnalyzeRequest):
        record = projects.load_project(projects_dir, slug)
        if record is None or not record.get("video"):
            return JSONResponse({"error": "Add a video first."}, status_code=400)
        try:
            job_id = pipeline.start_job(projects_dir, slug, fps=req.fps,
                                        max_usd=req.max_usd)
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:300]}"},
                                status_code=400)
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        status = pipeline.job_status(job_id)
        if status is None:
            return JSONResponse({"error": "No such job."}, status_code=404)
        if status.get("state") in ("done", "failed"):
            # the project's files have changed underneath any cached reader
            d = str(projects.project_dir(projects_dir, status["slug"]))
            if d in views:
                views[d].invalidate()
        return status

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel(job_id: str):
        return {"cancelled": pipeline.cancel_job(job_id)}

    # ---- reading a bundle (project or launch bundle) ----------------------

    @app.get("/api/bundle")
    def api_bundle(project: str = None):
        v = view_for(project)
        if not v.ready():
            return JSONResponse({"error": "This project has not been analyzed yet."},
                                status_code=409)
        v._load()
        base = f"/api/projects/{project}/bundle-media" if project else "/media"
        return {**v.meta, "video_url": f"{base}/{v.meta['video_file']}"}

    @app.get("/api/projects/{slug}/bundle-media/{filename}")
    def api_bundle_media(slug: str, filename: str):
        """A project's analyzed video. Name-checked rather than joined blindly,
        so a crafted filename cannot walk out of the project directory."""
        if "/" in filename or "\\" in filename or filename.startswith("."):
            return JSONResponse({"error": "bad filename"}, status_code=400)
        path = os.path.join(str(projects.project_dir(projects_dir, slug)), filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    @app.get("/api/tracks")
    def api_tracks(project: str = None):
        v = view_for(project)
        v._load()
        con = duckdb.connect(v.db_path, read_only=True)
        rows = con.execute(
            "SELECT track_id, frame, t, x, y FROM tracks ORDER BY frame, track_id"
        ).fetchall()
        con.close()
        return [{"track_id": r[0], "frame": r[1], "t": r[2], "x": r[3], "y": r[4]}
                for r in rows]

    @app.get("/api/events")
    def api_events(project: str = None):
        v = view_for(project)
        v._load()
        con = duckdb.connect(v.db_path, read_only=True)
        rows = con.execute("""
            SELECT event_id, type, actor_ids, frame_start, frame_end, zone, value
            FROM events ORDER BY frame_start
        """).fetchdf()
        con.close()
        rows = rows.astype(object).where(rows.notna(), None)
        out = []
        for r in rows.to_dict(orient="records"):
            out.append({
                **r,
                "label": _event_label(r),
                "citation": v.cite(r["frame_start"], r["frame_end"])
                if r["frame_start"] is not None else None,
            })
        return out

    @app.get("/api/report")
    def api_report(project: str = None):
        v = view_for(project)
        v._load()
        statements = [
            {"text": s["text"], "citations": v.fact_citations(s["citations"])}
            for s in v.statements
        ]
        return {"statements": statements, "fact_sheet": v.fact_sheet,
                "uncalibrated": bool(v.meta.get("uncalibrated"))}

    @app.post("/api/chat")
    def api_chat(req: ChatRequest):
        v = view_for(req.project)
        v._load()
        result = chat_backend.answer(req.question, v.fact_sheet)
        return {"answer": result["answer"],
                "citations": v.fact_citations(result["citations"])}

    # ---- promptable search ------------------------------------------------

    @app.get("/api/searches")
    def api_searches(project: str = None):
        v = view_for(project)
        return {"searches": list_searches(v.dir),
                "spent_usd": round(spent["usd"], 4),
                "session_cap_usd": SEARCH_MAX_USD_PER_SESSION}

    @app.post("/api/search")
    def api_search(req: SearchRequest):
        """Run a typed concept over this clip's frames and return cited hits.

        A cached concept is returned without sending anything, so the ceiling is
        only ever charged for genuinely new work, and it is checked before the
        call rather than after.
        """
        concept = (req.concept or "").strip()
        if not concept:
            return {"error": "Type something to search for."}
        v = view_for(req.project)

        cached = None if req.refresh else load_cached(v.dir, concept, req.stride)
        if cached is None and spent["usd"] >= SEARCH_MAX_USD_PER_SESSION:
            return {"error": (
                f"This session has spent ${spent['usd']:.2f} of its "
                f"${SEARCH_MAX_USD_PER_SESSION:.2f} search budget. Restart the "
                "server to search for something new, or ask for a concept "
                "already run.")}

        try:
            result = concept_search(
                v.dir, concept, stride=req.stride, max_frames=DEFAULT_MAX_FRAMES,
                max_usd=SEARCH_MAX_USD_PER_REQUEST, use_cache=not req.refresh)
        except SearchError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:300]}"}

        if not result.get("cached"):
            spent["usd"] += float(result.get("cost_usd") or 0.0)
        return {**result, "session_spent_usd": round(spent["usd"], 4)}

    return app


def main():
    parser = argparse.ArgumentParser(description="Proxemix")
    parser.add_argument("--bundle", default=None,
                        help="open one pre-built bundle directly, skipping projects")
    parser.add_argument("--projects", default=DEFAULT_PROJECTS_DIR)
    parser.add_argument("--demo", action="store_true",
                        help="open --bundle directly, skipping the project manager")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    app = create_app(args.bundle, projects_dir=args.projects,
                     auto_open_bundle=args.demo)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
