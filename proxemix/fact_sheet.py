"""Stage 8a — the fact sheet: a fixed Python pass over the DuckDB tables
that emits every fact the report/export layers are allowed to cite, each
one born with its own frame range and a fact ID. No LLM touches this
file; `report_writer.py`'s stub (and later Muse Wonder) may only cite
fact IDs that exist here, which is what `report_verifier.py` enforces.
"""
import json

from .timecode import frame_citation, frame_to_seconds

CATEGORY_ZONE = "zone"
CATEGORY_DENSITY = "density"
CATEGORY_CONTACT = "contact"
CATEGORY_OUTLIER = "outlier"
CATEGORY_OD = "od"
CATEGORY_GROUP = "group"
CATEGORY_PROXEMICS = "proxemics"


def _fact(fid, category, claim, frame_start, frame_end, metrics):
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    return {
        "fact_id": fid,
        "category": category,
        "claim": claim,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "t_start_s": frame_to_seconds(frame_start),
        "t_end_s": frame_to_seconds(frame_end),
        "citation": frame_citation(frame_start, frame_end),
        "metrics": metrics,
    }


def _busiest_zone_facts(con, counter):
    rows = con.execute("""
        SELECT zone, sum(value) AS total_dwell_s, count(*) AS n_visits,
               count(DISTINCT actor_ids) AS n_tracks,
               min(frame_start) AS fs, max(frame_end) AS fe
        FROM events WHERE type = 'zone_dwell' GROUP BY zone
        ORDER BY total_dwell_s DESC
    """).fetchdf()
    facts = []
    if rows.empty:
        return facts
    top = rows.iloc[0]
    facts.append(_fact(
        next(counter), CATEGORY_ZONE,
        f"'{top.zone}' is the busiest zone by total dwell time: {top.total_dwell_s:,.1f}s "
        f"across {int(top.n_visits)} visits from {int(top.n_tracks)} distinct tracks.",
        top.fs, top.fe,
        {"zone": top.zone, "total_dwell_s": float(top.total_dwell_s),
         "n_visits": int(top.n_visits), "n_tracks": int(top.n_tracks)},
    ))
    return facts


def _densest_minute_fact(con, counter):
    per_minute = con.execute("""
        SELECT CAST(t / 60 AS INTEGER) AS minute,
               min(frame) AS fs, max(frame) AS fe,
               avg(median_module_m2) AS mean_median_module,
               min(median_module_m2) AS worst_module,
               max(n_people) AS peak_people
        FROM density_timeline GROUP BY minute ORDER BY mean_median_module ASC LIMIT 1
    """).fetchdf()
    if per_minute.empty:
        return []
    r = per_minute.iloc[0]
    grade_row = con.execute("""
        SELECT fruin_grade FROM density_timeline
        WHERE CAST(t / 60 AS INTEGER) = ? ORDER BY median_module_m2 ASC LIMIT 1
    """, [int(r.minute)]).fetchone()
    grade = grade_row[0] if grade_row else None
    return [_fact(
        next(counter), CATEGORY_DENSITY,
        f"Minute {int(r.minute)} is the densest sustained window in the dataset: median module "
        f"dips to {r.worst_module:.2f} m^2/person (Fruin grade {grade}), with up to "
        f"{int(r.peak_people)} people in frame.",
        r.fs, r.fe,
        {"minute": int(r.minute), "worst_median_module_m2": float(r.worst_module),
         "fruin_grade": grade, "peak_people": int(r.peak_people)},
    )]


def _longest_contact_fact(con, counter):
    row = con.execute("""
        SELECT actor_ids, frame_start, frame_end, value FROM events
        WHERE type = 'contact' ORDER BY value DESC LIMIT 1
    """).fetchone()
    if row is None:
        return []
    actor_ids, fs, fe, duration = row
    return [_fact(
        next(counter), CATEGORY_CONTACT,
        f"The longest sustained contact episode is between tracks {actor_ids}, "
        f"lasting {duration:,.1f}s.",
        fs, fe, {"actor_ids": actor_ids, "duration_s": float(duration)},
    )]


def _top_outlier_facts(con, counter, top_n=5):
    rows = con.execute("""
        SELECT o.track_id, o.strangeness_score, o.worst_feature, o.worst_feature_z, o.reason,
               s.first_frame, s.last_frame
        FROM track_outliers o JOIN track_summary s ON o.track_id = s.track_id
        ORDER BY o.strangeness_score DESC LIMIT ?
    """, [top_n]).fetchdf()
    facts = []
    for _, r in rows.iterrows():
        facts.append(_fact(
            next(counter), CATEGORY_OUTLIER,
            f"Track {r.track_id} is a statistical outlier ({r.reason}).",
            r.first_frame, r.last_frame,
            {"track_id": r.track_id, "strangeness_score": float(r.strangeness_score),
             "worst_feature": r.worst_feature, "worst_feature_z": float(r.worst_feature_z),
             "reason": r.reason},
        ))
    return facts


def _od_highlight_facts(con, counter):
    facts = []
    common = con.execute("""
        SELECT origin_cell, dest_cell, count(*) AS n FROM od_table
        GROUP BY origin_cell, dest_cell ORDER BY n DESC LIMIT 1
    """).fetchone()
    if common is not None:
        origin, dest, n = common
        span = con.execute("SELECT min(first_frame), max(last_frame) FROM track_summary").fetchone()
        facts.append(_fact(
            next(counter), CATEGORY_OD,
            f"The most common origin-destination pair on the OD grid is {origin} -> {dest}, "
            f"taken by {int(n)} tracks.",
            span[0], span[1], {"origin_cell": origin, "dest_cell": dest, "n_tracks": int(n)},
        ))

    least_direct = con.execute("""
        SELECT o.track_id, o.directness_ratio, o.path_dist_m, o.straight_dist_m,
               s.first_frame, s.last_frame
        FROM od_table o JOIN track_summary s ON o.track_id = s.track_id
        WHERE o.directness_ratio IS NOT NULL ORDER BY o.directness_ratio ASC LIMIT 1
    """).fetchone()
    if least_direct is not None:
        track_id, ratio, path_m, straight_m, fs, fe = least_direct
        facts.append(_fact(
            next(counter), CATEGORY_OD,
            f"Track {track_id} has the least direct path in the dataset: directness ratio "
            f"{ratio:.4f} ({path_m:,.1f}m of path to cover {straight_m:,.1f}m of net displacement).",
            fs, fe, {"track_id": track_id, "directness_ratio": float(ratio),
                     "path_dist_m": float(path_m), "straight_dist_m": float(straight_m)},
        ))
    return facts


def _largest_group_fact(con, counter):
    row = con.execute("""
        SELECT actor_ids, frame_start, frame_end, value FROM events
        WHERE type = 'group' ORDER BY value DESC LIMIT 1
    """).fetchone()
    if row is None:
        return []
    actor_ids, fs, fe, n_members = row
    return [_fact(
        next(counter), CATEGORY_GROUP,
        f"The largest persistent group cluster has {int(n_members)} members, sustained from "
        f"frame {int(fs):,} to {int(fe):,}.",
        fs, fe, {"actor_ids": actor_ids, "n_members": int(n_members)},
    )]


def _hall_band_breakdown_fact(con, counter):
    rows = con.execute("""
        SELECT zone AS band, sum(value) AS total_s, count(*) AS n_episodes,
               min(frame_start) AS fs, max(frame_end) AS fe
        FROM events WHERE type = 'hall_band' GROUP BY band
    """).fetchdf()
    if rows.empty:
        return []
    fs, fe = int(rows.fs.min()), int(rows.fe.max())
    breakdown = {r.band: {"total_s": float(r.total_s), "n_episodes": int(r.n_episodes)}
                 for _, r in rows.iterrows()}
    parts = ", ".join(f"{k}={v['total_s']:,.0f}s" for k, v in sorted(breakdown.items()))
    return [_fact(
        next(counter), CATEGORY_PROXEMICS,
        f"Across all dyads, Hall-band occupancy totals: {parts}.",
        fs, fe, {"breakdown": breakdown},
    )]


def _zone_entry_latency_facts(con, counter):
    facts = []
    for label, order in [("fastest", "ASC"), ("slowest", "DESC")]:
        row = con.execute(f"""
            SELECT actor_ids, zone, frame_start, frame_end, value FROM events
            WHERE type = 'zone_first_entry' ORDER BY value {order} LIMIT 1
        """).fetchone()
        if row is None:
            continue
        actor_ids, zone, fs, fe, latency = row
        facts.append(_fact(
            next(counter), CATEGORY_ZONE,
            f"Track {actor_ids}'s {label} zone entry is '{zone}', reached {latency:,.1f}s "
            f"after it was first observed.",
            fs, fe, {"track_id": actor_ids, "zone": zone, "latency_s": float(latency)},
        ))
    return facts


def build_fact_sheet(con):
    """Returns a list of fact dicts, each with a unique fact_id, a frame
    range, and the raw metrics behind its claim. Order is deterministic.
    """
    counter = (f"F{i:03d}" for i in range(1, 10_000))
    facts = []
    facts += _busiest_zone_facts(con, counter)
    facts += _densest_minute_fact(con, counter)
    facts += _longest_contact_fact(con, counter)
    facts += _top_outlier_facts(con, counter)
    facts += _od_highlight_facts(con, counter)
    facts += _largest_group_fact(con, counter)
    facts += _hall_band_breakdown_fact(con, counter)
    facts += _zone_entry_latency_facts(con, counter)
    return facts


def write_fact_sheet(con, out_path):
    facts = build_fact_sheet(con)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    return facts
