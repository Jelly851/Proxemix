"""Stage 4 - calibration: image pixels -> floor-plane meters.

Takes 4 image points and their floor-plane correspondences (or one known
distance, assuming the 4 points form a rectangle) from a config dict/JSON
file, builds a cv2 homography, and projects track x/y into meters.

Config schema (JSON), one of two ways to give the destination rectangle:

    {
        "image_points": [[x,y], [x,y], [x,y], [x,y]],   # nw, ne, se, sw order
        "world_points_m": [[x,y], [x,y], [x,y], [x,y]]  # exact correspondences
    }

or the "one known distance" shortcut, assuming image_points trace a
rectangle in reading order (nw, ne, se, sw) of the given size:

    {
        "image_points": [[x,y], [x,y], [x,y], [x,y]],
        "rect_size_m": [width, height]
    }
    {
        "image_points": [[x,y], [x,y], [x,y], [x,y]],
        "known_distance_m": 12.5   # square rect_size_m = [d, d]
    }

The module itself is exact given exact inputs; real-world accuracy depends
entirely on how good the supplied points/distances are.
"""
import json

import cv2
import numpy as np


def load_calibration_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _world_points(config):
    if "world_points_m" in config:
        return np.array(config["world_points_m"], dtype=np.float64)
    if "rect_size_m" in config:
        w, h = config["rect_size_m"]
    elif "known_distance_m" in config:
        w = h = config["known_distance_m"]
    else:
        raise ValueError(
            "calibration config needs one of: world_points_m, rect_size_m, known_distance_m"
        )
    # nw, ne, se, sw, matching the image_points reading order
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


def build_homography(config):
    """Returns (H, world_points) where H maps image pixel (x,y,1) -> world (x,y) meters."""
    image_pts = np.array(config["image_points"], dtype=np.float64)
    if image_pts.shape != (4, 2):
        raise ValueError(f"calibration needs exactly 4 image_points, got {image_pts.shape}")
    world_pts = _world_points(config)
    H, _ = cv2.findHomography(image_pts, world_pts, method=0)
    if H is None:
        raise ValueError("cv2.findHomography failed on the given points — check they aren't collinear")
    return H, world_pts


def project_points(H, xy):
    """xy: (N, 2) array of pixel coords. Returns (N, 2) array of meter coords."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(xy, H)
    return out.reshape(-1, 2)


def apply_calibration_to_tracks(df, config):
    """Adds x_m, y_m (projected floor-plane meters) and vx_m, vy_m (finite
    difference of the meter positions on each track's own frame grid) to a
    copy of `df`. Does not touch the original pixel x/y/vx/vy columns.
    """
    H, _ = build_homography(config)
    out = df.copy()
    meters = project_points(H, out[["x", "y"]].to_numpy())
    out["x_m"] = meters[:, 0]
    out["y_m"] = meters[:, 1]

    out = out.sort_values(["track_id", "frame"])
    vx_m = np.zeros(len(out))
    vy_m = np.zeros(len(out))
    for _, idx in out.groupby("track_id", sort=False).indices.items():
        idx = np.asarray(idx)
        if idx.size < 2:
            continue
        sub = out.iloc[idx]
        dt = np.gradient(sub["t"].to_numpy())
        dt[dt == 0] = np.nan
        vx = np.gradient(sub["x_m"].to_numpy()) / dt
        vy = np.gradient(sub["y_m"].to_numpy()) / dt
        vx_m[idx] = np.nan_to_num(vx)
        vy_m[idx] = np.nan_to_num(vy)
    out["vx_m"] = vx_m
    out["vy_m"] = vy_m
    return out


def self_test():
    """Exact round-trip check: a known affine image->world mapping should
    recover with ~0 error. Not run as part of normal imports.
    """
    true_scale = 0.02  # meters per pixel, pure scale, no rotation/perspective
    image_pts = np.array([[100, 100], [900, 100], [900, 700], [100, 700]], dtype=np.float64)
    world_pts = image_pts * true_scale
    H, _ = cv2.findHomography(image_pts, world_pts, method=0)
    probe = np.array([[500.0, 400.0], [0.0, 0.0], [1920.0, 1080.0]])
    got = project_points(H, probe)
    expect = probe * true_scale
    err = np.abs(got - expect).max()
    assert err < 1e-6, f"calibration self-test failed, max error {err}"
    print(f"calibration self-test passed, max error {err:.2e} m")


if __name__ == "__main__":
    self_test()
