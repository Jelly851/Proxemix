"""Voronoi density + Fruin Level-of-Service grades over time.

Per frame: each present person's "module" (m^2 of floor space effectively
theirs) is their Voronoi cell's area, bounded by the calibrated floor
rectangle (unbounded cells at the crowd's edge are clipped to it, per the
standard Steffen & Seyfried 2010 bounded-Voronoi density method). Fruin
(1971) LOS grades a module into A (most free) through F (most congested).
"""
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

FRUIN_BANDS = [
    (3.25, float("inf"), "A"),
    (2.30, 3.25, "B"),
    (1.40, 2.30, "C"),
    (0.90, 1.40, "D"),
    (0.50, 0.90, "E"),
    (0.00, 0.50, "F"),
]


def fruin_grade(module_m2):
    for lo, hi, grade in FRUIN_BANDS:
        if lo <= module_m2 < hi:
            return grade
    return "F"


def _voronoi_finite_polygons_2d(vor, radius):
    """Reconstructs infinite Voronoi regions as large-but-finite polygons.
    Standard recipe (scipy cookbook) for 2D bounded Voronoi diagrams.
    Returns {point_index: [vertex_xy, ...]} in CCW order.
    """
    new_regions = {}
    center = vor.points.mean(axis=0)

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1 in range(len(vor.points)):
        region_idx = vor.point_region[p1]
        region = vor.regions[region_idx]
        if region and all(v >= 0 for v in region):
            new_regions[p1] = [vor.vertices[v] for v in region]
            continue

        ridges = all_ridges.get(p1, [])
        new_vertices_idx = [v for v in region if v >= 0]
        new_points = [vor.vertices[v] for v in new_vertices_idx]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # finite ridge, already included via region
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            normal = np.array([-t[1], t[0]])
            midpoint = (vor.points[p1] + vor.points[p2]) / 2.0
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_points.append(far_point)

        pts = np.array(new_points)
        if len(pts) < 3:
            new_regions[p1] = None
            continue
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(angles)]
        new_regions[p1] = pts.tolist()
    return new_regions


def compute_frame_modules(xy, bbox):
    """xy: (N,2) meter coords for one frame. bbox: (xmin,ymin,xmax,ymax) —
    the calibrated floor rectangle, used to clip unbounded cells. Returns
    an (N,) array of module areas (m^2 per person), NaN where N < 4 (too
    few points for a meaningful Voronoi tessellation that frame).
    """
    n = len(xy)
    if n < 4:
        return np.full(n, np.nan)
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    vor = Voronoi(xy)
    regions = _voronoi_finite_polygons_2d(vor, radius=span * 10)
    clip_box = box(*bbox)
    areas = np.full(n, np.nan)
    for i, verts in regions.items():
        if verts is None:
            continue
        poly = Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(clip_box)
        areas[i] = clipped.area
    return areas


def density_timeline(df, bbox, x_col="x_m", y_col="y_m"):
    """df: calibrated track rows. Returns a per-frame summary: frame, t,
    n_people, mean_module_m2, median_module_m2, min_module_m2, fruin_grade
    (graded off the MEDIAN module, i.e. the typical pedestrian's space that
    frame — a single mean/worst-case grade per frame, not per person).
    """
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        xy = g[[x_col, y_col]].to_numpy()
        modules = compute_frame_modules(xy, bbox)
        valid = modules[~np.isnan(modules)]
        if valid.size == 0:
            continue
        median_m = float(np.median(valid))
        rows.append({
            "frame": int(frame), "t": float(g["t"].iloc[0]), "n_people": len(g),
            "mean_module_m2": float(np.mean(valid)),
            "median_module_m2": median_m,
            "min_module_m2": float(np.min(valid)),
            "fruin_grade": fruin_grade(median_m),
        })
    return pd.DataFrame(rows)
