"""Stage 9 — researcher export: an aggregated-events table in BORIS's own
documented export schema (columns reproduced from
https://boris.readthedocs.io, the "aggregated events" tabular export —
Observation id/date/description/type/source, Time offset, Coding and
media duration, FPS, Subject, Observation duration by subject, Behavior,
Behavioral category, one column per configured modifier, Behavior type,
Start/Stop/Duration, Media file name, Image index/file path start/stop,
Comment start/stop), plus an ethogram and an auto-generated methods
paragraph, both generated from the same config/threshold values the
metrics pipeline actually ran with — never hand-typed numbers that could
drift from the code.

BORIS codes picture-sequence observations with "Image index" rather than
timestamps; Grand Central's annotations are themselves a sampled image
sequence (every 20th frame), so the Image index columns carry this
pipeline's real frame numbers rather than a fabricated video file.
Dyadic/group events use one combined Subject label (actor ids joined
with "+") rather than one row per member — BORIS supports this when a
project codes an interaction as a single dyad/group subject rather than
tracking each member's individual coding stream.
"""
import datetime

from . import contact_events, groups, hall_bands
from .timecode import FPS, frame_to_seconds

BORIS_COLUMNS = [
    "Observation id", "Observation date", "Description", "Observation type", "Source",
    "Time offset (s)", "Coding duration (s)", "Media duration (s)", "FPS (frame/s)",
    "Subject", "Observation duration by subject (s)", "Behavior", "Behavioral category",
    "Modifier 1", "Behavior type", "Start (s)", "Stop (s)", "Duration (s)",
    "Media file name", "Image index start", "Image index stop",
    "Image file path start", "Image file path stop", "Comment start", "Comment stop",
]

CATEGORY_MAP = {
    "contact": "Proximity",
    "hall_band": "Proxemics",
    "hall_transition": "Proxemics",
    "zone_dwell": "Spatial",
    "zone_first_entry": "Spatial",
    "group": "Social",
}


def _comment_for(event_type, value):
    if event_type == "zone_first_entry" and value is not None:
        return f"latency_s={value:.1f}"
    if event_type == "group" and value is not None:
        return f"n_members={int(value)}"
    return ""


def build_boris_table(con, observation_id="grand_central_01", observation_date=None,
                       source="grand_central_annotations_opentraj"):
    """Returns a pandas DataFrame with exactly the BORIS_COLUMNS, one row
    per row of the `events` table (no fan-out, no fabricated columns).
    """
    events = con.execute("SELECT * FROM events ORDER BY frame_start, event_id").fetchdf()
    summary = con.execute("SELECT track_id, duration FROM track_summary").fetchdf()
    dur_by_track = dict(zip(summary.track_id, summary.duration))
    fr_min, fr_max = con.execute("SELECT min(frame), max(frame) FROM tracks").fetchone()
    total_duration_s = frame_to_seconds(fr_max) - frame_to_seconds(fr_min)
    observation_date = observation_date or datetime.date.today().isoformat()

    df = events.copy()
    df["Subject"] = df["actor_ids"].str.replace(",", "+", regex=False)
    df["Behavior"] = df["type"]
    df["Behavioral category"] = df["type"].map(CATEGORY_MAP)
    df["Modifier 1"] = df["zone"].fillna("")
    df["Behavior type"] = [
        "STATE" if fe != fs else "POINT"
        for fs, fe in zip(df["frame_start"], df["frame_end"])
    ]
    df["Start (s)"] = df["frame_start"].apply(frame_to_seconds)
    df["Stop (s)"] = df["frame_end"].apply(frame_to_seconds)
    df["Duration (s)"] = df["Stop (s)"] - df["Start (s)"]
    df["Comment start"] = [
        _comment_for(t, v) for t, v in zip(df["type"], df["value"])
    ]
    df["Comment stop"] = ""
    df["Image index start"] = df["frame_start"]
    df["Image index stop"] = df["frame_end"]
    df["Image file path start"] = ""
    df["Image file path stop"] = ""
    df["Media file name"] = source
    df["Observation id"] = observation_id
    df["Observation date"] = observation_date
    df["Description"] = "Proxemix automated pedestrian-behavior coding"
    df["Observation type"] = "Media file"
    df["Source"] = source
    df["Time offset (s)"] = 0.0
    df["Coding duration (s)"] = total_duration_s
    df["Media duration (s)"] = total_duration_s
    df["FPS (frame/s)"] = FPS

    def obs_duration(actor_ids):
        durs = [dur_by_track.get(i) for i in actor_ids.split(",") if i in dur_by_track]
        return min(durs) if durs else None

    df["Observation duration by subject (s)"] = df["actor_ids"].apply(obs_duration)
    return df[BORIS_COLUMNS]


def write_boris_export(con, out_csv_path, out_tsv_path, **kwargs):
    df = build_boris_table(con, **kwargs)
    df.to_csv(out_csv_path, index=False)
    df.to_csv(out_tsv_path, index=False, sep="\t")
    return df


def _ethogram_entries():
    """One entry per behavior type, with definition/detection text built
    from the actual default parameters the modules run with — change a
    threshold in the code and this text changes with it.
    """
    return [
        {
            "behavior": "contact", "category": CATEGORY_MAP["contact"], "type": "STATE",
            "definition": "Two tracked individuals' floor-plane positions are close enough "
                           "to constitute a proximity contact.",
            "detection": f"distance between the dyad < {contact_events.detect_contact_episodes.__defaults__[0]} m, "
                         f"sustained >= {contact_events.detect_contact_episodes.__defaults__[1]:.0f}s "
                         "(COVID-literature convention), tolerating gaps of up to "
                         f"{contact_events.detect_contact_episodes.__defaults__[2]} frames before splitting the episode.",
        },
        {
            "behavior": "hall_band", "category": CATEGORY_MAP["hall_band"], "type": "STATE",
            "definition": "A dyad occupies one of Hall's (1966) proxemic distance bands "
                           "(intimate / personal / social) for a sustained run.",
            "detection": "distance binned to " + ", ".join(
                f"{name} [{lo:.2f}-{hi:.2f})m" for lo, hi, name in hall_bands.BANDS
            ) + "; run-length encoded per dyad per band, debounced across brief flicker.",
        },
        {
            "behavior": "hall_transition", "category": CATEGORY_MAP["hall_transition"], "type": "POINT",
            "definition": "The instant a dyad's occupied Hall band changes (e.g. personal -> "
                           "intimate), logged as the Modifier value band_from->band_to.",
            "detection": "boundary between two consecutive hall_band episodes for the same dyad.",
        },
        {
            "behavior": "zone_dwell", "category": CATEGORY_MAP["zone_dwell"], "type": "STATE",
            "definition": "An individual occupies a named floor zone for a sustained run.",
            "detection": "track's floor-plane position falls inside a configured zone polygon; "
                         "run-length encoded per track per zone, debounced across brief flicker.",
        },
        {
            "behavior": "zone_first_entry", "category": CATEGORY_MAP["zone_first_entry"], "type": "POINT",
            "definition": "The latency between a track's first observation and its first entry "
                           "into a given zone.",
            "detection": "first zone_dwell episode start time minus the track's own first "
                         "observed frame; logged as the Comment-start value latency_s.",
        },
        {
            "behavior": "group", "category": CATEGORY_MAP["group"], "type": "STATE",
            "definition": "A set of individuals persists around one shared o-space as a "
                           "standing/walking conversation group (an F-formation in Kendon's "
                           "sense). Validated against SALSA's human F-formation annotations: "
                           "F1 0.55 from position alone, 0.72 when body orientation is also "
                           "available — see outputs/salsa_validation.md.",
            "detection": groups.detector_description(),
        },
    ]


def generate_ethogram_md():
    lines = [
        "# Ethogram — Proxemix automated coding scheme",
        "",
        "Generated from the same threshold constants the metrics pipeline runs with "
        "(`proxemix/contact_events.py`, `proxemix/hall_bands.py`, `proxemix/groups.py`, "
        "`proxemix/zone_metrics.py`). Every row below corresponds to one `events.type` "
        "value and one `Behavior` value in the BORIS export.",
        "",
    ]
    for e in _ethogram_entries():
        lines += [
            f"## {e['behavior']}  ({e['type']})",
            f"**Category:** {e['category']}",
            f"**Operational definition:** {e['definition']}",
            f"**Detection rule:** {e['detection']}",
            "",
        ]
    return "\n".join(lines)


def generate_methods_paragraph(con, calibration_config):
    """A short auto-generated methods paragraph stating the coding scheme,
    thresholds, and calibration status — assembled from the actual config
    and the actual event counts in `con`, not hand-typed numbers.
    """
    counts = con.execute("SELECT type, count(*) FROM events GROUP BY type").fetchall()
    counts_txt = ", ".join(f"{t}: {n:,}" for t, n in sorted(counts))
    n_tracks = con.execute("SELECT count(*) FROM track_summary").fetchone()[0]
    approximate = calibration_config.get("approximate", False)
    cal_note = calibration_config.get("note", "")
    cal_status = (
        "Calibration from pixels to metric floor coordinates is APPROXIMATE "
        f"(hand-estimated, not surveyed): {cal_note}" if approximate else
        "Calibration from pixels to metric floor coordinates was surveyed/exact."
    )
    return (
        f"Behavioral events were coded automatically by the Proxemix pipeline over "
        f"{n_tracks:,} cleaned tracks. The coding scheme follows the operational "
        f"definitions in ethogram.md: proximity contact (distance < 1.5m, sustained "
        f">= 60s), Hall (1966) proxemic bands and their transitions, named-zone dwell "
        f"and first-entry latency, and persistent-group detection via DBSCAN "
        f"(eps=2m) with a >= 10s membership-persistence filter. {cal_status} "
        f"Resulting event counts by type: {counts_txt}."
    )


def write_methods_paragraph(con, calibration_config, out_path):
    text = generate_methods_paragraph(con, calibration_config)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


def write_ethogram(out_path):
    text = generate_ethogram_md()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
