"""End-to-end acceptance check for the keyless Proxemix core: ingest ->
Grand Central adapter (stands in for SAM) -> track cleaning -> storage.
Prints each stage's counts. Run with: py smoke_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from proxemix import ingest, storage, track_cleaning
from proxemix.calibration import apply_calibration_to_tracks, load_calibration_config
from proxemix.grand_central_adapter import load_grand_central_tracks, make_synthetic_fixture
from proxemix.metrics_pipeline import run_metrics

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxemix", "configs")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
GC_ANNOTATION_DIR = os.path.join(DATA_DIR, "gc_annotations")


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("STAGE 1 — ingest (synthetic test video, 10s testsrc -> 5fps frames)")
    video_path = os.path.join(DATA_DIR, "test_video", "synthetic.mp4")
    work_dir = os.path.join(DATA_DIR, "test_video", "work")
    if not os.path.exists(video_path):
        import subprocess
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=640x480:rate=30",
            video_path,
        ], check=True, capture_output=True)
    t0 = time.time()
    manifest_path = ingest.run_ingest(video_path, work_dir)
    import json
    manifest = json.load(open(manifest_path))
    print(f"frames extracted: {manifest['frame_count']} (expected ~50 at 5fps over 10s)")
    print(f"manifest: {manifest_path}  ({time.time()-t0:.1f}s)")

    section("STAGE: Grand Central adapter (stands in for SAM output)")
    if os.path.isdir(GC_ANNOTATION_DIR) and os.listdir(GC_ANNOTATION_DIR):
        print(f"using real Grand Central annotations at {GC_ANNOTATION_DIR}")
        raw_tracks = load_grand_central_tracks(GC_ANNOTATION_DIR)
    else:
        print(f"!! real Grand Central data not found at {GC_ANNOTATION_DIR}")
        print("!! falling back to a synthetic fixture to prove the adapter's parsing")
        fixture_dir = os.path.join(DATA_DIR, "synthetic_gc_fixture")
        make_synthetic_fixture(fixture_dir)
        raw_tracks = load_grand_central_tracks(fixture_dir)
    print(f"raw rows: {len(raw_tracks)}, raw tracks: {raw_tracks.track_id.nunique()}")

    section("STAGE 3 — track cleaning")
    t0 = time.time()
    cleaned, report = track_cleaning.clean_tracks(raw_tracks)
    print(f"cleaning took {time.time()-t0:.1f}s")
    for k, v in report.items():
        print(f"  {k}: {v}")

    section("STAGE 6 — storage (DuckDB)")
    db_path = os.path.join(DATA_DIR, "proxemix.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)
    counts = storage.store_cleaned_tracks(db_path, cleaned)
    print(f"db: {db_path}")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")

    section("STAGE 4 — calibration (pixels -> meters, Grand Central approximate config)")
    cal_cfg = load_calibration_config(os.path.join(CONFIG_DIR, "calibration_gc.json"))
    t0 = time.time()
    calibrated = apply_calibration_to_tracks(cleaned, cal_cfg)
    print(f"calibration took {time.time()-t0:.1f}s")
    print(f"  x_m range: {calibrated.x_m.min():.1f} .. {calibrated.x_m.max():.1f}")
    print(f"  y_m range: {calibrated.y_m.min():.1f} .. {calibrated.y_m.max():.1f}")

    section("STAGE 5 — events and metrics (contact, Hall bands, zone dwell, density/Fruin, OD, groups, outliers)")
    con = storage.init_db(db_path)
    zones_cfg_path = os.path.join(CONFIG_DIR, "zones_gc.json")
    t0 = time.time()
    event_counts = run_metrics(con, calibrated, cal_cfg, zones_cfg_path)
    print(f"metrics pipeline took {time.time()-t0:.1f}s")
    for k, v in sorted(event_counts.items()):
        print(f"  {k}: {v}")
    total_events = con.execute("SELECT count(*) FROM events").fetchone()[0]
    print(f"events table total: {total_events} rows")
    con.close()

    section("STAGE 8 — fact sheet, stub report draft, mechanical verifier")
    from proxemix.fact_sheet import write_fact_sheet
    from proxemix.report_verifier import verify_and_render
    from proxemix.report_writer import StubReportWriter

    outputs_dir = os.path.join(HERE, "outputs")
    report_dir = os.path.join(outputs_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    con = storage.init_db(db_path)
    facts = write_fact_sheet(con, os.path.join(report_dir, "fact_sheet.json"))
    print(f"fact sheet: {len(facts)} facts -> {os.path.join(report_dir, 'fact_sheet.json')}")

    draft = StubReportWriter().draft(facts)
    poisoned_statements = list(draft["statements"])
    poisoned_statements.insert(1, {
        "text": "An invented, never-computed claim the verifier must catch.",
        "citations": ["F999_DOES_NOT_EXIST"],
    })
    poisoned_draft = {"statements": poisoned_statements}
    clean_statements, dropped, markdown = verify_and_render(poisoned_draft, facts)
    print(f"verifier proof: fed 1 invented citation (F999_DOES_NOT_EXIST), dropped: {dropped}")
    assert dropped == [(1, "F999_DOES_NOT_EXIST")], "verifier failed to drop the invented citation"
    assert all("F999_DOES_NOT_EXIST" not in s["citations"] for s in clean_statements)
    assert "never-computed claim" not in markdown, "a statement with no surviving citation must not render"

    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"report: {len(clean_statements)} statements ({len(draft['statements'])} genuine + "
          f"1 poisoned, now dropped) -> {report_path}")

    section("STAGE 9 — researcher export (BORIS schema) and IE export (xlsx + spaghetti)")
    from proxemix.exports_ie import write_ie_workbook, write_spaghetti_svg
    from proxemix.exports_researcher import (
        write_boris_export, write_ethogram, write_methods_paragraph,
    )

    researcher_dir = os.path.join(outputs_dir, "researcher")
    os.makedirs(researcher_dir, exist_ok=True)
    boris_df = write_boris_export(
        con, os.path.join(researcher_dir, "events.csv"), os.path.join(researcher_dir, "events.tsv"))
    print(f"BORIS export: {len(boris_df)} rows -> researcher/events.csv, researcher/events.tsv")
    write_ethogram(os.path.join(researcher_dir, "ethogram.md"))
    write_methods_paragraph(con, cal_cfg, os.path.join(researcher_dir, "methods.md"))
    print(f"ethogram.md and methods.md -> {researcher_dir}")

    import csv
    valid_pairs = set(con.execute("SELECT frame_start, frame_end FROM events").fetchall())
    n_rows, n_bad = 0, 0
    with open(os.path.join(researcher_dir, "events.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            fs, fe = int(row["Image index start"]), int(row["Image index stop"])
            if (fs, fe) not in valid_pairs:
                n_bad += 1
    print(f"CSV round-trip check: {n_rows} rows parsed back, {n_bad} with a frame range "
          "not present in the events table")
    assert n_bad == 0, "BORIS export has frame ranges that don't exist in the events table"

    ie_dir = os.path.join(outputs_dir, "ie")
    os.makedirs(ie_dir, exist_ok=True)
    write_ie_workbook(con, os.path.join(ie_dir, "ie_workbook.xlsx"))
    write_spaghetti_svg(con, cal_cfg, zones_cfg_path, os.path.join(ie_dir, "spaghetti.svg"))
    print(f"IE workbook (5 sheets) + spaghetti diagram -> {ie_dir}")
    con.close()

    section("STAGE 10 — demo UI (bundle build + API, no browser here; see ui_test.py)")
    from fastapi.testclient import TestClient

    from proxemix.demo_bundle import build_demo_bundle
    from proxemix.serve import create_app

    demo_bundle_dir = os.path.join(DATA_DIR, "demo_bundle")
    bundle_meta = build_demo_bundle(demo_bundle_dir)
    print(f"demo bundle: {demo_bundle_dir}")
    print(f"  event_counts: {bundle_meta['event_counts']}")
    print(f"  facts: {bundle_meta['n_facts']}")

    app = create_app(demo_bundle_dir)
    with TestClient(app) as client:
        r = client.get("/api/bundle")
        assert r.status_code == 200 and r.json()["fps"] == 25.0, "bundle meta endpoint failed"

        r = client.get("/api/tracks")
        assert r.status_code == 200 and len(r.json()) > 0, "tracks endpoint returned nothing"

        r = client.get("/api/events")
        events = r.json()
        assert r.status_code == 200 and len(events) > 0, "events endpoint returned nothing"
        assert all(e.get("label") for e in events), "every event needs a display label"

        r = client.get("/api/report")
        report = r.json()
        assert r.status_code == 200 and len(report["statements"]) > 0, "report endpoint returned nothing"
        assert any(s["citations"] for s in report["statements"]), "report has no cited statements"

        r = client.post("/api/chat", json={"question": "busiest zone?"})
        chat = r.json()
        assert r.status_code == 200 and chat["citations"], "chat stub failed to answer a basic question"

        r = client.get("/")
        assert r.status_code == 200 and "proxemix" in r.text.lower(), "index page did not load"
    print(f"UI API checks passed: {len(events)} events, {len(report['statements'])} report statements, "
          f"chat answered with {len(chat['citations'])} citation(s)")

    section("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
