"""Stage 5 orchestrator: runs every event/metric module over a calibrated,
zoned track table and lands the results in DuckDB — event rows (who,
where, start, end) in the existing `events` table, summary tables
(density timeline, OD matrix, work-sampling / track features, outlier
ranking) alongside it. Pattern everywhere: boolean predicate per frame ->
debounce -> run-length encode -> event rows.
"""
import duckdb
import pandas as pd

from . import (contact_events, density_fruin, facing, groups, hall_bands, od_path,
               outliers, zone_metrics)
from .calibration import build_homography, load_calibration_config
from .pairwise import compute_pairwise_distances
from .zones import add_zone_column, load_zones_config

SUMMARY_SCHEMA = {
    "density_timeline": """
        CREATE TABLE IF NOT EXISTS density_timeline (
            frame BIGINT, t DOUBLE, n_people BIGINT,
            mean_module_m2 DOUBLE, median_module_m2 DOUBLE, min_module_m2 DOUBLE,
            fruin_grade VARCHAR
        )
    """,
    "od_table": """
        CREATE TABLE IF NOT EXISTS od_table (
            track_id VARCHAR, origin_cell VARCHAR, dest_cell VARCHAR,
            straight_dist_m DOUBLE, path_dist_m DOUBLE, directness_ratio DOUBLE
        )
    """,
    "track_outliers": """
        CREATE TABLE IF NOT EXISTS track_outliers (
            track_id VARCHAR, duration_s DOUBLE, distance_m DOUBLE,
            mean_speed_m_s DOUBLE, max_speed_m_s DOUBLE, directness_ratio DOUBLE,
            n_points BIGINT, strangeness_score DOUBLE, worst_feature VARCHAR,
            worst_feature_z DOUBLE, reason VARCHAR
        )
    """,
}


def run_metrics(con, calibrated_tracks, calibration_config, zones_config,
                 contact_threshold_m=1.5, contact_min_duration_s=60.0,
                 pairwise_max_radius_m=3.6, od_grid_cells=4, outlier_top_n=25):
    """calibrated_tracks: the full calibrated (x_m/y_m/vx_m/vy_m) track
    table. Returns a dict of per-event-type counts, and writes everything
    to `con` (events table + the summary tables above).
    """
    zones = load_zones_config(zones_config)
    zoned = add_zone_column(calibrated_tracks, zones)

    _, world_pts = build_homography(calibration_config)
    bbox = (0.0, 0.0, float(world_pts[:, 0].max()), float(world_pts[:, 1].max()))

    pairwise = compute_pairwise_distances(calibrated_tracks, max_radius_m=pairwise_max_radius_m)

    contact_ep = contact_events.detect_contact_episodes(
        pairwise, distance_threshold_m=contact_threshold_m, min_duration_s=contact_min_duration_s)
    contact_rows = contact_events.episodes_to_event_rows(contact_ep)

    hall_ep, hall_trans = hall_bands.detect_hall_band_episodes(pairwise)
    hall_rows = hall_bands.episodes_to_event_rows(hall_ep)
    hall_trans_rows = hall_bands.transitions_to_event_rows(hall_trans)

    visits = zone_metrics.detect_zone_visits(zoned)
    visit_rows = zone_metrics.visits_to_event_rows(visits)
    first_t = calibrated_tracks.sort_values(["track_id", "frame"]).groupby("track_id")["t"].first()
    latency = zone_metrics.latency_to_first_entry(visits, first_t)
    latency_rows = zone_metrics.latency_to_event_rows(latency)

    # Facing first: the grouper's o-space test needs it, and on a transit scene
    # like Grand Central almost everybody is walking, so motion supplies it for
    # most person-frames. Stationary people come out honestly unknown and the
    # grouper falls back to distance plus its transitivity guard for them.
    tracks_facing = facing.add_facing(calibrated_tracks)
    clusters = groups.detect_frame_clusters(tracks_facing)
    group_ep = groups.detect_group_episodes(clusters)
    group_rows = groups.episodes_to_event_rows(group_ep)

    all_events = pd.concat(
        [contact_rows, hall_rows, hall_trans_rows, visit_rows, latency_rows, group_rows],
        ignore_index=True,
    )
    all_events.insert(0, "event_id", range(1, len(all_events) + 1))
    all_events = all_events[["event_id", "type", "actor_ids", "frame_start", "frame_end", "zone", "value"]]

    con.register("events_df", all_events)
    con.execute("DELETE FROM events")
    con.execute("INSERT INTO events SELECT * FROM events_df")

    density = density_fruin.density_timeline(calibrated_tracks, bbox)
    od = od_path.build_od_table(calibrated_tracks, bbox, n_cells=od_grid_cells)
    feats = outliers.build_track_features(calibrated_tracks, od)
    ranked = outliers.rank_outliers(feats, top_n=outlier_top_n)

    for name, ddl in SUMMARY_SCHEMA.items():
        con.execute(ddl)
        con.execute(f"DELETE FROM {name}")
    con.register("density_df", density)
    con.execute("INSERT INTO density_timeline SELECT * FROM density_df")
    con.register("od_df", od)
    con.execute("INSERT INTO od_table SELECT * FROM od_df")
    con.register("outliers_df", ranked[[
        "track_id", "duration_s", "distance_m", "mean_speed_m_s", "max_speed_m_s",
        "directness_ratio", "n_points", "strangeness_score", "worst_feature",
        "worst_feature_z", "reason",
    ]])
    con.execute("INSERT INTO track_outliers SELECT * FROM outliers_df")

    counts = all_events["type"].value_counts().to_dict()
    counts["density_timeline"] = len(density)
    counts["od_table"] = len(od)
    counts["track_outliers"] = len(ranked)
    counts["group_clusters_raw"] = len(clusters)
    return counts
