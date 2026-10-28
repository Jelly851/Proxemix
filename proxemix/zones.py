"""Zone definitions: polygons in floor-plane meters, loaded from a config
file. Zones are user-supplied in the real product; this module just loads
and tests membership against whatever polygons it is given.
"""
import json

import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon  # used only to validate the config's polygons


def load_zones_config(path):
    """Returns {name: matplotlib.path.Path} — Path.contains_points is a
    vectorized, numpy-fast membership test, which matters here since this
    runs over 600k+ track rows. shapely.Polygon is used only to validate
    each polygon is well-formed (non-self-intersecting) before that.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    zones = {}
    for z in cfg["zones"]:
        poly = Polygon(z["polygon_m"])
        if not poly.is_valid:
            raise ValueError(f"zone '{z['name']}' polygon is not valid (self-intersecting?)")
        zones[z["name"]] = Path(z["polygon_m"])
    return zones


def assign_zone(x_m, y_m, zones):
    """Vectorized: for arrays of x_m/y_m, returns an array of zone names
    (or None) — first matching zone wins if polygons overlap.
    """
    pts = np.column_stack([x_m, y_m])
    out = np.full(len(pts), None, dtype=object)
    remaining = np.ones(len(pts), dtype=bool)
    for name, path in zones.items():
        idx = np.where(remaining)[0]
        if idx.size == 0:
            break
        inside = path.contains_points(pts[idx])
        hit = idx[inside]
        out[hit] = name
        remaining[hit] = False
    return out


def add_zone_column(df, zones, x_col="x_m", y_col="y_m"):
    out = df.copy()
    out["zone"] = assign_zone(out[x_col].to_numpy(), out[y_col].to_numpy(), zones)
    return out
