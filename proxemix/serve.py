"""Stage 10 — the demo UI backend. FastAPI serves one static page (the
video player + canvas overlay + event list + report + chat panel) and a
small JSON API over a bundle directory (video.mp4 + proxemix.duckdb).

Run: py -m proxemix.serve --bundle data/demo_bundle
Builds the synthetic demo bundle automatically on first run if the given
directory doesn't exist yet. Point --bundle at any other directory built
the same way (see readme.md "Pointing the UI at a new bundle") to demo
real footage with zero code changes.
"""
import argparse
import json
import os

import duckdb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chat_stub import StubChatBackend
from .demo_bundle import build_demo_bundle
from .timecode import frame_citation

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")

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


class ChatRequest(BaseModel):
    question: str


def create_app(bundle_dir):
    if not os.path.isdir(bundle_dir) or not os.path.exists(os.path.join(bundle_dir, "bundle_meta.json")):
        build_demo_bundle(bundle_dir)

    with open(os.path.join(bundle_dir, "bundle_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(bundle_dir, "fact_sheet.json"), "r", encoding="utf-8") as f:
        fact_sheet = json.load(f)
    with open(os.path.join(bundle_dir, "report_statements.json"), "r", encoding="utf-8") as f:
        report_statements = json.load(f)
    facts_by_id = {f["fact_id"]: f for f in fact_sheet}

    db_path = os.path.join(bundle_dir, meta["db_file"])
    chat_backend = StubChatBackend()

    app = FastAPI(title="Proxemix demo UI")
    app.mount("/media", StaticFiles(directory=bundle_dir), name="media")
    app.mount("/static", StaticFiles(directory=os.path.join(WEBUI_DIR, "static")), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEBUI_DIR, "index.html"))

    @app.get("/api/bundle")
    def api_bundle():
        return {**meta, "video_url": f"/media/{meta['video_file']}"}

    @app.get("/api/tracks")
    def api_tracks():
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute("SELECT track_id, frame, t, x, y FROM tracks ORDER BY frame, track_id").fetchall()
        con.close()
        return [{"track_id": r[0], "frame": r[1], "t": r[2], "x": r[3], "y": r[4]} for r in rows]

    @app.get("/api/events")
    def api_events():
        con = duckdb.connect(db_path, read_only=True)
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
                "citation": frame_citation(r["frame_start"], r["frame_end"])
                if r["frame_start"] is not None else None,
            })
        return out

    @app.get("/api/report")
    def api_report():
        statements = []
        for s in report_statements:
            citations = [
                {"fact_id": cid, "frame_start": facts_by_id[cid]["frame_start"],
                 "frame_end": facts_by_id[cid]["frame_end"], "citation": facts_by_id[cid]["citation"]}
                for cid in s["citations"] if cid in facts_by_id
            ]
            statements.append({"text": s["text"], "citations": citations})
        return {"statements": statements, "fact_sheet": fact_sheet}

    @app.post("/api/chat")
    def api_chat(req: ChatRequest):
        result = chat_backend.answer(req.question, fact_sheet)
        citations = [
            {"fact_id": cid, "frame_start": facts_by_id[cid]["frame_start"],
             "frame_end": facts_by_id[cid]["frame_end"], "citation": facts_by_id[cid]["citation"]}
            for cid in result["citations"] if cid in facts_by_id
        ]
        return {"answer": result["answer"], "citations": citations}

    return app


def main():
    parser = argparse.ArgumentParser(description="Proxemix demo UI server")
    parser.add_argument("--bundle", default="data/demo_bundle", help="bundle directory (video.mp4 + proxemix.duckdb)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    app = create_app(args.bundle)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
