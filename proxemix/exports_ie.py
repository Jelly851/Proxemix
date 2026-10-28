"""Stage 9 — industrial-engineering export: an .xlsx workbook shaped like
Timer Pro's element-times sheet (Rating/Allowance columns present and
deliberately BLANK — those are IE judgment, faking them kills trust),
plus an ASME flow-process chart, a from-to zone matrix, a work-sampling
table computed from a 100% sample (continuous tracking, not random
spot-checks), and a Fruin Level-of-Service section. A spaghetti diagram
is written separately as an SVG (`write_spaghetti_svg`) since it is a
figure, not a worksheet.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .calibration import build_homography, project_points
from .timecode import frame_to_seconds
from .zones import load_zones_config

HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
BLANK_FILL = PatternFill("solid", fgColor="FFF2CC")
HEADER_FONT = Font(bold=True)


def _write_header(ws, headers, blank_cols=()):
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = HEADER_FONT
        cell.fill = BLANK_FILL if (c - 1) in blank_cols else HEADER_FILL
    for c in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16


def _add_element_times_sheet(wb, con, sample_per_zone=50):
    ws = wb.create_sheet("Element Times")
    headers = ["Element (zone)", "Track", "Start (s)", "Stop (s)", "Observed Time (s)",
               "Rating", "Allowance", "Normal Time (s)", "Standard Time (s)"]
    _write_header(ws, headers, blank_cols=(5, 6))  # Rating, Allowance columns flagged blank
    ws.cell(row=1, column=6).comment = None
    dwell = con.execute("""
        SELECT zone, actor_ids AS track, frame_start, frame_end, value AS observed_s
        FROM events WHERE type = 'zone_dwell' ORDER BY zone, value DESC
    """).fetchdf()
    rows = []
    for zone, g in dwell.groupby("zone"):
        rows.append(g.head(sample_per_zone))
    sampled = pd.concat(rows, ignore_index=True) if rows else dwell.head(0)

    r = 2
    for row in sampled.itertuples(index=False):
        ws.cell(row=r, column=1, value=row.zone)
        ws.cell(row=r, column=2, value=row.track)
        ws.cell(row=r, column=3, value=round(frame_to_seconds(row.frame_start), 2))
        ws.cell(row=r, column=4, value=round(frame_to_seconds(row.frame_end), 2))
        ws.cell(row=r, column=5, value=round(row.observed_s, 2))
        # Rating (F) / Allowance (G) left blank — IE judgment, never faked.
        ws.cell(row=r, column=8, value=f'=IF(F{r}="","",E{r}*F{r})')
        ws.cell(row=r, column=9, value=f'=IF(OR(F{r}="",G{r}=""),"",H{r}*(1+G{r}))')
        r += 1
    ws.freeze_panes = "A2"
    note = ws.cell(row=1, column=11,
                    value=f"Downsampled to {sample_per_zone} longest observations per zone "
                          f"element for readability ({len(sampled)} of "
                          f"{len(dwell)} total zone_dwell observations shown). "
                          "Rating/Allowance are blank by design — fill them in to "
                          "auto-compute Normal/Standard Time.")
    note.font = Font(italic=True)
    return ws


def _representative_track(con):
    row = con.execute("""
        SELECT actor_ids, count(*) AS n FROM events WHERE type = 'zone_dwell'
        GROUP BY actor_ids ORDER BY n DESC LIMIT 1
    """).fetchone()
    return row[0] if row else None


def _add_asme_flow_sheet(wb, con, track_id=None):
    ws = wb.create_sheet("ASME Flow Process Chart")
    track_id = track_id or _representative_track(con)
    headers = ["Step #", "ASME Symbol", "Activity", "Zone / Location", "Distance (m)", "Time (s)"]
    _write_header(ws, headers)
    ws.cell(row=1, column=8, value=f"Representative track {track_id} "
            "(most zone-to-zone transitions in the dataset).").font = Font(italic=True)
    if track_id is None:
        return ws

    visits = con.execute("""
        SELECT zone, frame_start, frame_end, value AS dwell_s FROM events
        WHERE type = 'zone_dwell' AND actor_ids = ? ORDER BY frame_start
    """, [track_id]).fetchdf()
    # calibrated (x_m/y_m) coords aren't persisted to `tracks`, so distance here
    # is pixel path length — a shape proxy for the flow chart, not a metric one.
    path = con.execute(
        "SELECT frame, x, y FROM tracks WHERE track_id = ? ORDER BY frame", [track_id]
    ).fetchdf()

    def path_distance_px(f0, f1):
        seg = path[(path.frame >= f0) & (path.frame <= f1)]
        if len(seg) < 2:
            return 0.0
        d = np.diff(seg[["x", "y"]].to_numpy(), axis=0)
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    r = 2
    step = 1
    prev_end = None
    SYMBOLS = {"operation": "○", "transport": "⇒", "delay": "D"}
    for row in visits.itertuples(index=False):
        if prev_end is not None and row.frame_start > prev_end:
            dist_px = path_distance_px(prev_end, row.frame_start)
            ws.cell(row=r, column=1, value=step)
            ws.cell(row=r, column=2, value=SYMBOLS["transport"])
            ws.cell(row=r, column=3, value="Transport (open floor)")
            ws.cell(row=r, column=4, value="-")
            ws.cell(row=r, column=5, value=f"{dist_px:.0f} px (uncalibrated path)")
            ws.cell(row=r, column=6, value=round(frame_to_seconds(row.frame_start - prev_end), 2))
            r += 1
            step += 1
        symbol = SYMBOLS["delay"] if row.dwell_s > 300 else SYMBOLS["operation"]
        label = "Delay (extended dwell)" if row.dwell_s > 300 else "Operation (dwell)"
        ws.cell(row=r, column=1, value=step)
        ws.cell(row=r, column=2, value=symbol)
        ws.cell(row=r, column=3, value=label)
        ws.cell(row=r, column=4, value=row.zone)
        ws.cell(row=r, column=5, value="-")
        ws.cell(row=r, column=6, value=round(row.dwell_s, 2))
        r += 1
        step += 1
        prev_end = row.frame_end
    return ws


def _add_from_to_matrix_sheet(wb, con):
    ws = wb.create_sheet("From-To Matrix")
    visits = con.execute("""
        SELECT actor_ids AS track, zone, frame_start FROM events
        WHERE type = 'zone_dwell' ORDER BY track, frame_start
    """).fetchdf()
    pairs = []
    for _, g in visits.groupby("track"):
        zones = g["zone"].tolist()
        pairs += list(zip(zones[:-1], zones[1:]))
    if pairs:
        mat = pd.crosstab(
            pd.Series([p[0] for p in pairs], name="from"),
            pd.Series([p[1] for p in pairs], name="to"),
        )
    else:
        mat = pd.DataFrame()

    ws.cell(row=1, column=1, value="From \\ To").font = HEADER_FONT
    for c, col in enumerate(mat.columns, start=2):
        ws.cell(row=1, column=c, value=col).font = HEADER_FONT
    for r, (idx, rowvals) in enumerate(mat.iterrows(), start=2):
        ws.cell(row=r, column=1, value=idx).font = HEADER_FONT
        for c, col in enumerate(mat.columns, start=2):
            ws.cell(row=r, column=c, value=int(rowvals[col]))
    note = ws.cell(row=1, column=len(mat.columns) + 3,
                    value="Counts of consecutive zone-to-zone transitions "
                          "(a track's next named-zone visit after leaving the current one); "
                          f"{len(pairs)} transitions across {visits['track'].nunique()} tracks.")
    note.font = Font(italic=True)
    return ws


def _add_work_sampling_sheet(wb, con):
    ws = wb.create_sheet("Work Sampling")
    headers = ["Activity", "Person-Seconds Observed", "% of Total Observed Time", "N Visits"]
    _write_header(ws, headers)
    zone_time = con.execute("""
        SELECT zone, sum(value) AS person_s, count(*) AS n FROM events
        WHERE type = 'zone_dwell' GROUP BY zone ORDER BY person_s DESC
    """).fetchdf()
    total_person_s = con.execute("SELECT sum(duration) FROM track_summary").fetchone()[0]
    open_floor_s = total_person_s - zone_time["person_s"].sum()

    r = 2
    for row in zone_time.itertuples(index=False):
        ws.cell(row=r, column=1, value=row.zone)
        ws.cell(row=r, column=2, value=round(row.person_s, 1))
        ws.cell(row=r, column=3, value=round(100 * row.person_s / total_person_s, 2))
        ws.cell(row=r, column=4, value=int(row.n))
        r += 1
    ws.cell(row=r, column=1, value="open_floor (no zone)")
    ws.cell(row=r, column=2, value=round(open_floor_s, 1))
    ws.cell(row=r, column=3, value=round(100 * open_floor_s / total_person_s, 2))
    ws.cell(row=r, column=4, value="-")
    note = ws.cell(row=1, column=6,
                    value="100% sample: every tracked person-second in the clip, "
                          "not random spot-check snapshots (Tippett 1935 / ILO 1992 "
                          "work-sampling convention).")
    note.font = Font(italic=True)
    return ws


def _add_fruin_los_sheet(wb, con):
    ws = wb.create_sheet("Fruin LOS")
    headers = ["Fruin Grade", "N Samples", "% of Observed Samples",
               "Worst Median Module (m2/person)", "Frame Range (worst instance)"]
    _write_header(ws, headers)
    dist = con.execute("""
        SELECT fruin_grade, count(*) AS n FROM density_timeline GROUP BY fruin_grade
    """).fetchdf()
    total = dist["n"].sum()
    r = 2
    for grade in ["A", "B", "C", "D", "E", "F"]:
        row = dist[dist.fruin_grade == grade]
        n = int(row["n"].iloc[0]) if not row.empty else 0
        worst = con.execute("""
            SELECT median_module_m2, frame FROM density_timeline
            WHERE fruin_grade = ? ORDER BY median_module_m2 ASC LIMIT 1
        """, [grade]).fetchone()
        ws.cell(row=r, column=1, value=grade)
        ws.cell(row=r, column=2, value=n)
        ws.cell(row=r, column=3, value=round(100 * n / total, 2) if total else 0)
        ws.cell(row=r, column=4, value=round(worst[0], 2) if worst else "-")
        ws.cell(row=r, column=5, value=worst[1] if worst else "-")
        r += 1
    note = ws.cell(row=1, column=7,
                    value="Fruin (1971) walkway LOS, graded per sample off the frame's "
                          "median bounded-Voronoi module (Steffen & Seyfried 2010). "
                          "A = most free-flowing, F = most congested.")
    note.font = Font(italic=True)
    return ws


def write_ie_workbook(con, out_path, asme_track_id=None, sample_per_zone=50):
    wb = Workbook()
    wb.remove(wb.active)
    _add_element_times_sheet(wb, con, sample_per_zone=sample_per_zone)
    _add_asme_flow_sheet(wb, con, track_id=asme_track_id)
    _add_from_to_matrix_sheet(wb, con)
    _add_work_sampling_sheet(wb, con)
    _add_fruin_los_sheet(wb, con)
    wb.save(out_path)
    return out_path


def write_spaghetti_svg(con, calibration_config, zones_config_path, out_path,
                         max_tracks=150, seed=0):
    """All sampled tracks' paths drawn over the calibrated floor plane,
    zone polygons overlaid for context. Downsampled to `max_tracks` of
    however many tracks actually exist, stated on the figure itself —
    plotting all ~11.6k tracks would be unreadable ink soup.
    """
    H, world_pts = build_homography(calibration_config)
    bbox = (0.0, 0.0, float(world_pts[:, 0].max()), float(world_pts[:, 1].max()))

    all_tracks = con.execute("SELECT DISTINCT track_id FROM tracks").fetchdf()["track_id"].tolist()
    rng = np.random.default_rng(seed)
    n = min(max_tracks, len(all_tracks))
    chosen = sorted(rng.choice(all_tracks, size=n, replace=False).tolist())

    placeholders = ",".join("?" * len(chosen))
    rows = con.execute(
        f"SELECT track_id, x, y FROM tracks WHERE track_id IN ({placeholders}) "
        "ORDER BY track_id, frame", chosen,
    ).fetchdf()

    fig, ax = plt.subplots(figsize=(10, 10 * (bbox[3] / bbox[2])))
    for name, path in load_zones_config(zones_config_path).items():
        ax.add_patch(mpatches.Polygon(path.vertices, closed=True, fill=True,
                                       alpha=0.15, label=name, edgecolor="black", linewidth=0.5))
    for track_id, g in rows.groupby("track_id", sort=False):
        xy = project_points(H, g[["x", "y"]].to_numpy())
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.6, alpha=0.55)

    ax.set_xlim(bbox[0] - 2, bbox[2] + 2)
    ax.set_ylim(bbox[1] - 2, bbox[3] + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Spaghetti diagram — {n} of {len(all_tracks)} tracks sampled for readability "
                 f"(seed={seed}); zone polygons shaded for reference")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
