"""Origin-destination matrix and path efficiency (route directness ratio).

OD cells are a coarse grid over the calibrated floor rectangle (not the
named zones config) so every track contributes, including the ~88% of
Grand Central tracks here that never cross one of the four hand-picked
zones — a grid always has something to say about where a path started and
ended, which a handful of named zones alone would mostly leave blank.
"""
import numpy as np
import pandas as pd


def grid_cell_label(x, y, bbox, n_cells):
    xmin, ymin, xmax, ymax = bbox
    col = int(np.clip((x - xmin) / (xmax - xmin) * n_cells, 0, n_cells - 1))
    row = int(np.clip((y - ymin) / (ymax - ymin) * n_cells, 0, n_cells - 1))
    return f"r{row}c{col}"


def build_od_table(df, bbox, n_cells=4, x_col="x_m", y_col="y_m"):
    """Returns one row per track: track_id, origin_cell, dest_cell,
    straight_dist_m, path_dist_m, directness_ratio (straight/path, capped
    at 1.0; NaN for single-point tracks).
    """
    rows = []
    for track_id, g in df.sort_values("frame").groupby("track_id", sort=False):
        xy = g[[x_col, y_col]].to_numpy()
        origin = grid_cell_label(xy[0, 0], xy[0, 1], bbox, n_cells)
        dest = grid_cell_label(xy[-1, 0], xy[-1, 1], bbox, n_cells)
        straight = float(np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1]))
        if len(xy) >= 2:
            d = np.diff(xy, axis=0)
            path = float(np.hypot(d[:, 0], d[:, 1]).sum())
        else:
            path = 0.0
        ratio = min(straight / path, 1.0) if path > 0 else np.nan
        rows.append({
            "track_id": track_id, "origin_cell": origin, "dest_cell": dest,
            "straight_dist_m": straight, "path_dist_m": path, "directness_ratio": ratio,
        })
    return pd.DataFrame(rows)


def od_matrix(od_table):
    """Pivot to a cell x cell count matrix."""
    return pd.crosstab(od_table["origin_cell"], od_table["dest_cell"])
