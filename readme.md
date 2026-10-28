# Proxemix — keyless core

The parts of the Proxemix pipeline (see `Body/app/outputs/hackathon tests/proximix/build spec.md`)
that can be built and proven before the Meta SAM API key arrives: ingest (stage 1), track
cleaning (stage 3), calibration (stage 4), events/metrics (stage 5), storage (stage 6), the
fact sheet + stubbed report + mechanical verifier (stage 8), the researcher/IE exports
(stage 9), and the demo UI (stage 10) — plus a Grand Central adapter that stands in for SAM's
track output so every later stage has real data to run against.

No SAM key is used or required by anything in this folder.

## Setup

```
py -m pip install -r requirements.txt
```

Requires `ffmpeg` on PATH (used for stage-1 re-encoding).

## Modules

* `proxemix/ingest.py` — stage 1. `reencode()` runs ffmpeg with dense keyframes and no
  B-frames (`-g 12 -bf 0 -movflags +faststart`); `extract_frames()` pulls frames at a fixed
  fps (default 5) via PyAV's native timestamps (not OpenCV seeking, which is known to land
  frames off). `run_ingest()` chains both and writes `frames/manifest.json`.
* `proxemix/grand_central_adapter.py` — loads the published Grand Central Station pedestrian
  trajectories (OpenTraj format: one `.txt` file per track under `.../GC/Annotation/`, flat
  `x, y, frame` triples) into the canonical `(track_id, frame, x, y)` table every later stage
  expects, whoever produced the tracks. `make_synthetic_fixture()` writes a tiny fixture in
  the same format if the real annotation files are ever missing.
* `proxemix/salsa_adapter.py` — the same job for the SALSA dataset (FBK, CC BY-NC-SA 3.0),
  which unlike Grand Central also ships *human-annotated conversation groups*, so it can
  validate `groups.py` rather than just feed it. Loads `geometryGT/NN.csv` (positions already
  on the floor plane in metres — the module docstring records the three pieces of evidence for
  that, and `sanity_check()` re-derives them) and `fformationGT.csv` (the F-formation ground
  truth) for both parts, poster session and cocktail party. Provenance in `data/salsa/SOURCE.md`.
* `proxemix/salsa_validation.py` — scores `groups.py`'s detections against SALSA's human
  F-formation labels using the tolerant-match criterion from Setti et al. (PLOS ONE 2015), with
  one-to-one bipartite matching per frame, per-cardinality breakdowns, a failure-shape
  classification and a parameter sweep. `py -m proxemix.salsa_validation` (no flags — the
  shipped defaults are the scoring configuration). `--method dbscan` re-runs the pre-rebuild
  detector, `--facing oracle` feeds the grouper SALSA's own annotated body orientation to
  measure the upper bound, and the geometry thresholds are all flags. Result, every number and
  the honest failure analysis: `outputs/salsa_validation.md` — **F1 0.551 at the shipped
  defaults, 0.721 when body orientation is supplied**, against F1 0.83 for published methods
  that also use orientation (and 0.385 / 0.069 for the DBSCAN grouper this replaced).
* `proxemix/track_cleaning.py` — stage 3. Classifies each track's birth/death as `border`
  (real entry/exit at the image edge) or `mid_frame` (an occlusion artifact); bridges
  mid-frame gaps under ~2s by gating constant-velocity extrapolation against a data-driven
  max-speed threshold and solving the resulting local assignment problems with
  `scipy.optimize.linear_sum_assignment`; interpolates the remaining gaps on the data's own
  sample grid; Savitzky-Golay smooths x/y; finite-differences velocity.
* `proxemix/storage.py` — stage 6. DuckDB file with `tracks`, `events`, and `track_summary`.
* `proxemix/calibration.py` — stage 4. `build_homography()` takes 4 image points plus either
  their exact floor-plane meter correspondences, a known rectangle width/height, or one known
  distance (assumes a square), and calls `cv2.findHomography()`. `apply_calibration_to_tracks()`
  projects every track row's x/y into meters (`x_m`, `y_m`) via `cv2.perspectiveTransform()` and
  finite-differences `vx_m`/`vy_m` on each track's own frame grid. `self_test()` (run via
  `py -m proxemix.calibration`) proves the module exact given exact inputs: a synthetic pure-scale
  mapping round-trips to <1e-6 m error.
* `proxemix/zones.py` — loads a zones config (polygons in floor-plane meters) and assigns each
  track row to a zone (or none) via `matplotlib.path.Path.contains_points` (vectorized — this
  runs over 600k+ rows).
* `proxemix/pairwise.py` — per frame, `scipy.spatial.cKDTree.query_pairs()` finds every co-present
  pair within a radius, feeding both proximity modules below without an O(N^2) pass over everyone
  who was never actually near anyone.
* `proxemix/contact_events.py` — stage 5. Contact/proximity episodes per dyad: distance < 1.5m
  (configurable), debounced (brief flickers out of range don't split an episode), run-length
  encoded, kept if sustained >= 60s (configurable) — the COVID-literature convention named in the
  spec.
* `proxemix/hall_bands.py` — stage 5. Classifies each co-present pair's distance into Hall (1966)
  intimate/personal/social bands, same debounce+RLE pattern, plus the transitions between
  consecutive band-episodes per dyad. "Public" (>3.6m) isn't logged as its own stream — the
  pairwise search radius stops at 3.6m, since logging every pair that's merely "not near" is
  unbounded and not proxemically meaningful.
* `proxemix/zone_metrics.py` — stage 5. Dwell per zone, zone visits (debounce+RLE on "which zone,
  if any"), and latency to first entry (first visit's start time minus the track's own first
  observed frame).
* `proxemix/density_fruin.py` — stage 5. Per frame, each present person's "module" (m^2 of floor
  effectively theirs) is their Voronoi cell's area, bounded by the calibrated floor rectangle
  (the standard Steffen & Seyfried 2010 bounded-Voronoi density method); graded A-F per Fruin
  (1971) walkway level-of-service thresholds off the frame's median module.
* `proxemix/od_path.py` — stage 5. Origin-destination matrix over a coarse grid on the calibrated
  floor (not the named zones — most tracks never cross one of the four hand-picked zones, but a
  grid always has something to say), plus path efficiency (straight-line distance / actual path
  length) per track.
* `proxemix/facing.py` — stage 5 input. Which way each person is pointing, per track per
  frame: direction of travel, smoothed as a vector over ~1 s, and **withheld below 0.25 m/s**
  because a stationary person's motion direction is noise rather than facing (NaN angle, zero
  confidence, `facing_source="unknown"`). `apply_external_facing()` is the seam for the second
  source — SAM body keypoints, which is what gives a *standing* person a facing and which waits
  on the API key; it fills exactly the rows motion left unknown. Coverage measured: 70.8% of
  person-frames on Grand Central (a transit hall, everyone walking), 0.9-3.2% on SALSA (a
  standing party). `py -m proxemix.facing` self-tests both halves.
* `proxemix/groups.py` — stage 5. Conversation-group (F-formation) detection, rebuilt
  2026-10-26 on Kendon's definition rather than proximity alone: a candidate group must be
  connected within the 1.1 m linking radius, have *every* pair inside the size-scaled
  transitivity cap (the guard that stops one chain welding a room together, which single-linkage
  DBSCAN structurally cannot do), stand around a centre rather than along a line, and — wherever
  `facing.py` supplies a facing — converge on one shared o-space. Groups are built divisively
  (a failing set is cut along the weakest seam of its affinity graph and each half
  reconsidered); `method="dbscan"` keeps the old path selectable. Then debounce+RLE on the
  *exact* member set so a momentary co-location doesn't count as a group — it has to persist
  (>= 10s, configurable), and **the gap tolerance is in seconds, derived from the data's own
  sampling stride**, which is what the old `debounce_frames=40` could not be (it returned zero
  episodes on any dataset sampled more sparsely than Grand Central). Validated:
  `outputs/salsa_validation.md`. `py -m proxemix.groups` self-tests the chain, the bystander,
  the ring, the back-to-back veto and both sampling strides.
* `proxemix/outliers.py` — stage 5. Within-clip outlier triage: robust (median/MAD) z-scores
  across duration, distance, mean/max speed, and path directness; ranks tracks by worst-feature
  |z|, reason attached (e.g. "max_speed_m_s=35.89, z=104.5 vs the dataset's median track").
* `proxemix/metrics_pipeline.py` — stage 5 orchestrator. Estimates facing, then runs every
  module above over a calibrated, zoned track table; writes event rows (contact, hall_band, hall_transition,
  zone_dwell, zone_first_entry, group) into the existing `events` table, and summary tables
  (`density_timeline`, `od_table`, `track_outliers`) alongside it.
* `proxemix/timecode.py` — frame <-> time conversion shared by the fact sheet and both
  exports (`FPS = 25.0`, matching the Grand Central data; `frame_citation()` renders the
  spec's exact `[frame 4,210-4,540 / 02:48-03:02]` shape).
* `proxemix/fact_sheet.py` — stage 8a. A fixed Python pass over the DuckDB tables (no LLM)
  that emits every citable fact as structured JSON: busiest zone, densest minute + Fruin
  grade, longest contact, top-5 outliers with reasons, OD highlights, largest group, Hall-band
  totals, fastest/slowest zone entries. Every fact carries its own frame range and a sequential
  fact ID (`F001`, `F002`, ...).
* `proxemix/report_writer.py` — stage 8b. `ReportWriter` is the interface; `StubReportWriter`
  drafts narrative JSON deterministically (one statement per fact, citing that fact's ID, no
  model call) so everything downstream can be built today; `MuseWonderReportWriter` is where
  the real model plugs in once the hackathon key exists — same interface, currently raises
  `NotImplementedError`.
* `proxemix/report_verifier.py` — stage 8c, the spec's first "load-bearing wow mechanism."
  Drops any citation in a draft that doesn't match a fact ID in the fact sheet, then renders
  markdown with `[frame ...]` links for every citation that survived.
* `proxemix/exports_researcher.py` — stage 9, researcher export. `events.csv`/`.tsv` in
  BORIS's own documented "aggregated events" column schema (boris.readthedocs.io), an
  `ethogram.md` generated from the actual threshold constants the metrics modules run with
  (not hand-typed), and an auto-generated methods paragraph.
* `proxemix/exports_ie.py` — stage 9, IE export. An `.xlsx` workbook (element-times sheet with
  Rating/Allowance left blank, ASME flow-process chart, from-to zone matrix, work-sampling
  table, Fruin LOS section) plus a spaghetti-diagram SVG over the calibrated floor plane.
* `proxemix/demo_bundle.py` — stage 10 support. Builds a self-contained "bundle" directory
  (`video.mp4` + `proxemix.duckdb` + `fact_sheet.json` + `report_statements.json` +
  `bundle_meta.json`) by running 6 synthetic people through the *same* stage 3/4/5/6/8 pipeline
  every other dataset here uses — see "The demo bundle" below for why a synthetic fixture
  stands in for footage.
* `proxemix/chat_stub.py` — stage 10's chat panel backend. `ChatBackend` is the interface;
  `StubChatBackend` answers by keyword match against the fact sheet (no model call);
  `MuseWonderChatBackend` is where the real client plugs in later, same `answer()` signature.
* `proxemix/serve.py` — stage 10, the demo UI. FastAPI app serving one static page (video
  player + canvas overlay + event list + report + chat) and a small JSON API
  (`/api/bundle`, `/api/tracks`, `/api/events`, `/api/report`, `/api/chat`) over a bundle
  directory. `python -m proxemix.serve --bundle <dir>` builds the bundle on first run if it
  doesn't exist yet, then starts the server.
* `proxemix/webui/` — the static frontend: `index.html`, `static/app.js`, `static/style.css`.
  The video plays via a plain `<video>` tag; a same-sized `<canvas>` sits on top and redraws
  every real video frame via `requestVideoFrameCallback` (frame-exact in evergreen browsers,
  with a `timeupdate` fallback for browsers that lack it). Event rows, report citations, and
  chat citations all call the same `seekToFrame(frame)` function
  (`video.currentTime = (frame + 0.5) / fps`) — one handler, three entry points, per the spec.

## Proving it

Each piece below was run, not just written; commands reproduce the same run.

**Ingest**, on a synthetic 10s test video:
```
ffmpeg -y -f lavfi -i "testsrc=duration=10:size=640x480:rate=30" data/test_video/synthetic.mp4
py -c "from proxemix.ingest import run_ingest; print(run_ingest('data/test_video/synthetic.mp4', 'data/test_video/work'))"
```
Result: 50 frames (5 fps x 10s), manifest timestamps run 0.0 -> 9.8s.

**Grand Central adapter**, on the real published annotations:
```
py -c "from proxemix.grand_central_adapter import load_grand_central_tracks as L; df=L(r'...\Body\app\outputs\_tmp_gc_clone\datasets\GC\Annotation'); print(len(df), df.track_id.nunique())"
```
Result: 456,410 rows across 12,684 tracks. If that annotation folder is ever absent, the
adapter is instead proven against `make_synthetic_fixture()` (same code path, smaller data) —
the `smoke_test.py` fallback below demonstrates exactly that branch.

**Track cleaning**, on the Grand Central tracks:
```
py -c "
from proxemix.grand_central_adapter import load_grand_central_tracks as L
from proxemix.track_cleaning import clean_tracks
df = L(r'...\Body\app\outputs\_tmp_gc_clone\datasets\GC\Annotation')
cleaned, report = clean_tracks(df)
print(report)
"
```
Result: 12,684 tracks -> 11,595 after bridging 1,089 mid-frame gaps; 456,410 raw rows ->
633,998 rows after gap-filling on the data's native sample grid (177,588 of them
interpolated); no NaNs in x/y/vx/vy.

**Storage**:
```
py -c "
import pandas as pd
from proxemix.storage import store_cleaned_tracks
cleaned = pd.read_pickle('data/cache/cleaned_tracks.pkl')  # this pipeline's own local cache file — or just re-run clean_tracks()
print(store_cleaned_tracks('data/proxemix.duckdb', cleaned))
"
```
Result: `tracks` 633,998 rows, `track_summary` 11,595 rows (`events` is populated later, by the
stage-5 metrics pipeline below).

**Calibration**, on the cleaned Grand Central tracks, using `proxemix/configs/calibration_gc.json`:
```
py -c "
import pandas as pd
from proxemix.calibration import load_calibration_config, apply_calibration_to_tracks
cleaned = pd.read_pickle('data/cache/cleaned_tracks.pkl')
cfg = load_calibration_config('proxemix/configs/calibration_gc.json')
out = apply_calibration_to_tracks(cleaned, cfg)
print(out[['x_m','y_m','vx_m','vy_m']].describe())
"
py -m proxemix.calibration   # exact round-trip self-test, independent of any real data
```
Result: median pedestrian speed 1.50 m/s (a believable adult walking pace), floor-plane extent
roughly x -12 to 42m, y -32 to 47m (beyond the calibrated 34x45m rectangle at the edges, from
tracks that sit just outside the four calibration points — homography extrapolates past its
source quadrilateral, which is expected and is why the zones config only uses polygons well
inside that rectangle). Self-test: max error 7e-15 m on a synthetic exact mapping — the module
itself is exact given exact inputs.

**Zones**, assigning every row to one of the four configured Grand Central zones (or none):
```
py -c "
import pandas as pd
from proxemix.zones import load_zones_config, add_zone_column
calibrated = pd.read_pickle('data/cache/calibrated_tracks.pkl')  # from the calibration step above
zones = load_zones_config('proxemix/configs/zones_gc.json')
out = add_zone_column(calibrated, zones)
print(out['zone'].value_counts(dropna=False))
"
```
Result: ~88% of rows fall outside all four zones (open floor), the rest split across
`info_booth` (23,899), `east_doorway` (17,909), `west_doorway` (16,233), and
`east_side_corridor` (15,264).

**Events and metrics** (stage 5), run together by `proxemix/metrics_pipeline.py` over the
calibrated+zoned tracks — contact episodes, Hall-zone bands and transitions, zone dwell/visits/
latency, Voronoi density + Fruin grades, OD matrix + path efficiency, groups, outlier triage —
on the full Grand Central dataset (observed 131-240s across runs, machine-load dependent).
`smoke_test.py` below runs this automatically; see "Inspecting the extremes" for what the
numbers say.

**Fact sheet, stub report, and the mechanical verifier** (stage 8), run over the populated
`events`/`track_summary`/`density_timeline`/`od_table`/`track_outliers` tables:
```
py -c "
import duckdb
from proxemix.fact_sheet import write_fact_sheet
from proxemix.report_writer import StubReportWriter
from proxemix.report_verifier import verify_and_render
con = duckdb.connect('data/proxemix.duckdb', read_only=True)
facts = write_fact_sheet(con, 'outputs/report/fact_sheet.json')
draft = StubReportWriter().draft(facts)
draft['statements'].insert(1, {'text': 'An invented claim.', 'citations': ['F999_DOES_NOT_EXIST']})
clean, dropped, md = verify_and_render(draft, facts)
print('dropped:', dropped)
"
```
Result: 14 facts, each with its own frame range and fact ID. Feeding the verifier a draft with
one invented fact ID (`F999_DOES_NOT_EXIST`) drops exactly that citation and nothing else — the
statement's text survives, its citation link does not (`smoke_test.py` runs this as an
assertion, not just a print, so a regression here fails the acceptance check outright).

**Researcher and IE exports** (stage 9), over the same DuckDB file:
```
py -c "
import duckdb
from proxemix.calibration import load_calibration_config
from proxemix.exports_researcher import write_boris_export, write_ethogram, write_methods_paragraph
from proxemix.exports_ie import write_ie_workbook, write_spaghetti_svg
con = duckdb.connect('data/proxemix.duckdb', read_only=True)
write_boris_export(con, 'outputs/researcher/events.csv', 'outputs/researcher/events.tsv')
write_ethogram('outputs/researcher/ethogram.md')
cal_cfg = load_calibration_config('proxemix/configs/calibration_gc.json')
write_methods_paragraph(con, cal_cfg, 'outputs/researcher/methods.md')
write_ie_workbook(con, 'outputs/ie/ie_workbook.xlsx')
write_spaghetti_svg(con, cal_cfg, 'proxemix/configs/zones_gc.json', 'outputs/ie/spaghetti.svg')
"
```
Result: `events.csv`/`.tsv` (301,805 rows, one per `events` table row, in BORIS's own documented
aggregated-events column order), `ethogram.md` and `methods.md` generated from the running
code's own threshold constants, a 5-sheet `.xlsx` workbook, and an SVG spaghetti diagram of 150
sampled tracks over the calibrated floor plane with the four zones shaded for reference.

**Demo UI** (stage 10) — the bundle, the API, and the browser, each proven separately:
```
py -m proxemix.serve --bundle data/demo_bundle
```
Builds `data/demo_bundle/` on first run (ffmpeg generates a 640x480/25fps/12s test-pattern
clip; 6 synthetic people are run through track cleaning, calibration, zoning, stage-5 metrics,
the fact sheet, and the stub report — the real pipeline, not UI-only mock data), then serves
the page at `http://127.0.0.1:8000/`. Result: the player shows the test-pattern video with 6
colored dots + track-ID labels tracking in sync; the event list, report citations, and chat
answers all jump the video to the cited frame on click.

`smoke_test.py`'s stage 10 builds the same bundle and hits every API endpoint
(`/api/bundle`, `/api/tracks`, `/api/events`, `/api/report`, `/api/chat`, `/`) through FastAPI's
`TestClient` — no browser, asserting each one returns real data (events have display labels,
the report has cited statements, chat answers "busiest zone?" with a citation). Result: 48
events, 15 report statements, chat answered with 1 citation.

`py ui_test.py` is the actual browser proof (separate from the smoke test on purpose — it needs
Playwright + a real Chromium, and launches the server as its own subprocess on port 8731, killed
in a `finally` block whether or not the assertions pass). It clicks an event row and asserts
`video.currentTime` moved to that event's exact frame, asks the chat stub a question and clicks
its citation button, asserting the video seeks there too, and screenshots the player at each
step. Result: both seeks landed within 0.05s of the expected
`(frame + 0.5) / fps`; 3 screenshots in `outputs/ui/` (`01_initial_player.png`,
`02_event_seek_overlay.png` — 6 labeled dots visible mid-clip, `03_chat_citation_seek.png`).

## The demo bundle

A "bundle" is any directory containing `video.mp4`, a `proxemix.duckdb` with `tracks` (pixel
`x`/`y`) and `events` tables (plus `track_summary`, `density_timeline`, `od_table`,
`track_outliers` if the report/fact-sheet endpoints are to work), `fact_sheet.json`, and
`report_statements.json`, written as `bundle_meta.json` describes
(`fps`, `width`, `height`, `duration_s`, `video_file`, `db_file`). `proxemix/serve.py` only ever
reads that shape — it has no idea the included bundle is synthetic.

Grand Central (every other stage's dataset) has **no underlying video file**, only published
trajectory annotations, so it cannot be the demo bundle. The demo instead reuses the synthetic
ingest test video's own format (640x480, re-encoded with dense keyframes, same `ffmpeg testsrc`
pattern `smoke_test.py`'s stage 1 already proves ingest against) and generates a small
track/event fixture sized to match it — 6 people, 12 seconds, designed to exercise every event
type: two approach and stand close (contact + Hall-band occupancy/transition + a persistent
group), one dwells in `left_zone` the whole clip (busiest-zone + fastest zone-entry facts), one
walks into `right_zone` partway through (slowest zone-entry contrast + a distinct OD pair), one
crosses fast (speed outlier), and one circles in place (near-zero net displacement — a
directness-ratio outlier, the same artifact class documented in the real Grand Central data
below). `proxemix/demo_bundle.py`'s module docstring and `_track_specs()` comments name exactly
which mechanism each person is there to exercise.

### Pointing the UI at a new bundle

Nothing about the UI is synthetic-data-specific. To demo real footage:
1. Build a directory with real `video.mp4`.
2. Run that footage's tracks (from SAM once the key exists, or any adapter shaped like
   `grand_central_adapter.py`) through `track_cleaning.clean_tracks()` →
   `storage.store_cleaned_tracks()` → `calibration.apply_calibration_to_tracks()` →
   `metrics_pipeline.run_metrics()` → `fact_sheet.write_fact_sheet()` →
   `report_writer.StubReportWriter().draft()` → `report_verifier.verify_and_render()` — exactly
   what `demo_bundle.build_demo_bundle()` does for the synthetic fixture; copy that function as
   the template and swap in the real video/tracks.
3. Write `bundle_meta.json` with that footage's real `fps`/`width`/`height`/`duration_s`.
4. `py -m proxemix.serve --bundle <new dir>` — no changes to `serve.py` or `webui/` required.

**Whole chain, one command:**
```
py smoke_test.py
```
Runs ingest on the synthetic video, loads the real Grand Central tracks (or the synthetic
fixture if that data is ever missing), cleans them, stores them in DuckDB, calibrates to
meters, assigns zones, runs every stage-5 metric, builds the fact sheet, drafts and verifies a
stub report, and writes both exports — printing every stage's counts and where every file
landed. This is the acceptance check. Takes roughly 4-7 minutes end to end depending on machine
load (cleaning ~40-80s, metrics 131-240s, exports ~15s, the rest is seconds).

## What the approximate Grand Central calibration means for the numbers

`proxemix/configs/calibration_gc.json` is hand-estimated, not surveyed: Grand Central's main
concourse is publicly reported at roughly 275ft x 120ft (83.8m x 36.6m), and the four
calibration points pick a sub-region of that floor visible in the dataset's reference frame,
assigned a 34m x 45m span derived from that public figure and the image's own perspective —
not a measurement taken on site. Every meter-denominated number downstream (speeds, distances,
zone polygons, density modules, Fruin grades) inherits that same margin of error — treat them
as "plausible, roughly right," not survey-grade. The calibration *code* (`calibration.py`) is
exact given exact inputs, proven by its own synthetic self-test; swapping in four real
on-site-measured points (or a real known distance) would make every downstream number exact
too, with zero code changes.

## Known data artifacts

* One native (unmerged) Grand Central track (`006668`) has two points 20 frames apart with a
  large pixel jump (`vx` ~= -1252 px/frame, ~36 m/s) — present in the source annotation file
  itself, not introduced by cleaning. It surfaces on its own as the #6-ranked statistical
  outlier in `track_outliers` (reason: `max_speed_m_s=35.89, z=104.5`), which is the triage
  module doing its job rather than a new defect.
* Track `002125` is the single strangest track by the outlier ranking (`duration_s=3560.80,
  z=212.8`) — present for 3,560s of the dataset's ~4,800s span at a near-zero mean speed
  (0.007 m/s). Plausible as a long-dwelling person (a vendor, a busker) rather than an error,
  but unverifiable against ground truth beyond the trajectory itself — flagged as a lead, not
  a correction.
* The longest contact episode (dyad `010095`/`010096`, 893.6s / ~15 minutes, mean distance
  0.97m) checks out as plausible: both tracks span the exact same frame range and are each
  essentially stationary (median speed ~0.004 m/s) — two people standing near each other for
  fifteen minutes, not a stuck/duplicate pair (which would show near-zero *relative* distance
  with no real temporal overlap, not two independently-stationary tracks).
* The longest single zone dwell (track `004398`, `info_booth`, 2,839s / ~47 minutes, almost to
  the end of the dataset) is also plausible rather than a defect — near-zero speed the entire
  time, consistent with a booth attendant standing at their post rather than a passer-through.

## Inspecting the extremes

* **Densest minute's Fruin grade**: minute 28 of the clip (t ~1680-1740s) is the only sustained
  window that dips to grade C (median module as low as 1.78 m^2/person, 95-153 people in frame)
  — every other minute in the dataset stays at grade A or B.
* **Largest group — and the one finding the 2026-10-26 grouper rebuild overturned.** Under the
  old DBSCAN detector the largest group episode had **24 members** (t~1645-1659s, inside that
  same dense window), and this readme used to carry the warning that any episode above ~4-5
  members was "a dense moment," not a group: a flat 2 m radius merges everyone standing in a
  crowd into one blob. The rebuilt detector cannot do that — the transitivity guard refuses a
  set whose non-adjacent members are too far apart — and on the same data the largest episode
  is now **5 members** (`003807, 004150, 004151, 004152, 004156`, frames 47,460-47,800), with
  the 1,101 episodes distributed 972 pairs / 108 triples / 18 fours / 3 fives. The warning is
  withdrawn: group sizes can now be read as written. The detector's measured accuracy, with its
  own failure analysis, is `outputs/salsa_validation.md`.
* **Least direct path**: track `003346` has a directness ratio of 0.0015 — 154m of cumulative
  path length to end up 0.23m from where it started, in 85 seconds (which would require ~1.8
  m/s average speed with zero net displacement). This reads as a tracking-noise artifact
  (small-amplitude position jitter accumulating into a large path length), not a real
  pedestrian's movement — flagged here rather than smoothed away, same posture as the `006668`
  artifact above.

## Inspecting the exports

* **Fact sheet**: 14 facts, `F001` (busiest zone, `info_booth`, 18,080.8s total dwell) through
  `F014` (slowest zone entry, track `004410` into `east_doorway`, 2,028.8s latency). The
  outlier and group facts independently re-derive the same tracks already documented above
  (`002125`, the 24-member cluster, `003346`) straight from the DB, which cross-checks that
  documentation against a second, differently-written code path.
* **Verifier proof**: feeding `report_verifier.verify_and_render()` a draft with one invented
  fact ID (`F999_DOES_NOT_EXIST`) inserted among 15 genuine statements drops exactly that one
  citation — the statement's text renders in `report.md` with no frame link attached, every
  genuine statement keeps its link. `smoke_test.py` asserts this, not just prints it.
* **BORIS export**: 301,805 rows (one per `events` table row — no fan-out, no fabricated
  columns), round-trips cleanly through `csv.DictReader`, and every `(Image index start,
  Image index stop)` pair was checked against the live `events` table with zero mismatches.
* **IE workbook**: 5 sheets all load back via `openpyxl.load_workbook()`. Element Times sheet
  (200 sampled rows, 50 longest per zone) ranges 7.2s (`east_side_corridor`) to 2,839.2s
  (`info_booth`, track `004398` — the same 47-minute dwell flagged above); Rating/Allowance
  columns are genuinely blank, with `Normal/Standard Time` formulas that activate once a human
  IE fills them in. Work Sampling sheet's `info_booth` total (18,080.8 person-seconds) matches
  the fact sheet's busiest-zone fact exactly, cross-checking both.
* **Spaghetti SVG**: rasterized and eyeballed — 150 of 11,595 tracks (stated on the figure)
  drawn over the calibrated floor plane with all four zone polygons shaded for reference; paths
  read as plausible pedestrian flow (through-traffic lines plus a tight knot of short, jittery
  paths near one corner, consistent with someone standing still) rather than noise.

## Not built here

Everything genuinely not built yet: the LLM layer itself (stage 7 — `MuseWonderReportWriter`
and `MuseWonderChatBackend` are stubbed interfaces, not integrations) and stage 2 (SAM tracking
itself), which needs the Meta API key this folder deliberately does not depend on. Stages 8
(fact sheet, stub report, verifier), 9 (researcher + IE exports), and 10 (demo UI) are built
and proven above.
