"""Stage 6 — storage: DuckDB file with tracks / events / track_summary,
per the build spec.
"""
import duckdb
import numpy as np
import pandas as pd

SCHEMA = {
    "tracks": """
        CREATE TABLE IF NOT EXISTS tracks (
            track_id VARCHAR,
            frame BIGINT,
            t DOUBLE,
            x DOUBLE,
            y DOUBLE,
            vx DOUBLE,
            vy DOUBLE
        )
    """,
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            event_id BIGINT,
            type VARCHAR,
            actor_ids VARCHAR,
            frame_start BIGINT,
            frame_end BIGINT,
            zone VARCHAR,
            value DOUBLE
        )
    """,
    "track_summary": """
        CREATE TABLE IF NOT EXISTS track_summary (
            track_id VARCHAR,
            first_frame BIGINT,
            last_frame BIGINT,
            duration DOUBLE,
            distance DOUBLE,
            mean_speed DOUBLE
        )
    """,
}


def init_db(db_path):
    con = duckdb.connect(db_path)
    for ddl in SCHEMA.values():
        con.execute(ddl)
    return con


def build_track_summary(cleaned_df):
    rows = []
    for track_id, g in cleaned_df.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        d = np.diff(g[["x", "y"]].values, axis=0)
        distance = float(np.hypot(d[:, 0], d[:, 1]).sum()) if len(g) >= 2 else 0.0
        duration = float(g["t"].iloc[-1] - g["t"].iloc[0])
        mean_speed = distance / duration if duration > 0 else 0.0
        rows.append({
            "track_id": track_id,
            "first_frame": int(g["frame"].iloc[0]),
            "last_frame": int(g["frame"].iloc[-1]),
            "duration": duration,
            "distance": distance,
            "mean_speed": mean_speed,
        })
    return pd.DataFrame(rows)


def store_cleaned_tracks(db_path, cleaned_df):
    """Populate `tracks` and `track_summary` from a stage-3 cleaned table.
    `events` is created with its schema and left empty (no event detection
    stage exists yet). Returns row counts per table.
    """
    con = init_db(db_path)
    tracks_out = cleaned_df[["track_id", "frame", "t", "x", "y", "vx", "vy"]]
    con.register("tracks_df", tracks_out)
    con.execute("DELETE FROM tracks")
    con.execute("INSERT INTO tracks SELECT * FROM tracks_df")

    summary = build_track_summary(cleaned_df)
    con.register("summary_df", summary)
    con.execute("DELETE FROM track_summary")
    con.execute("INSERT INTO track_summary SELECT * FROM summary_df")

    counts = {
        "tracks": con.execute("SELECT count(*) FROM tracks").fetchone()[0],
        "events": con.execute("SELECT count(*) FROM events").fetchone()[0],
        "track_summary": con.execute("SELECT count(*) FROM track_summary").fetchone()[0],
    }
    con.close()
    return counts
