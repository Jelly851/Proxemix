"""SALSA's camera model — image pixels <-> the floor plane the annotations live on.

Why this is not `proxemix/calibration.py`
-----------------------------------------
`calibration.py` builds a 4-point homography from a config a human draws on a
frame. SALSA ships something better: a real calibrated camera per view, as
`data/salsa/calibration/camN_calib.txt` — a rotation matrix, a translation
vector, a focal length with per-axis pixel scales, a principal point, and two
radial distortion coefficients. Using the homography path would throw away the
distortion term (k1 = -0.31 on CAM1, which moves an edge pixel by tens of
pixels) and would need four hand-picked correspondences that do not exist.

So this module implements the camera directly, and `floor_homography()` hands
the rest of Proxemix an equivalent `cv2`-style 3x3 for the undistorted image,
so `calibration.project_points` and `sam_video._world_dirs` keep working
unchanged.

The convention, established rather than assumed
-----------------------------------------------
The file gives R and T. Two readings are possible: world-to-camera
(`Xc = R Xw + T`) or camera-to-world (`Xw = R Xc + T`). They are distinguished
by where they put the camera:

* world-to-camera puts the optical centre at `-R' T` = (-9.98, 1.27, -4.15) m,
  which is 14 m outside a room spanning x 3.9-14.8, y 0.1-6.7, and 4 m *below*
  the floor.
* camera-to-world puts it at `T` = (8.00, 6.79, 2.88) m — at the far edge of
  the room's y extent, 2.9 m up. A corner-mounted camera, 2.9 m high.

Only the second is a camera. The facing experiment then goes further and
projects the annotated floor positions of all 18 participants into real
extracted frames: if the convention were wrong the points would not land on
people, and they do. Z is up, in metres, same frame as `Ground_Position_X/Y`.

Pixel model (OpenCV's, with SALSA's per-axis scales):

    x_n = Xc/Zc,  y_n = Yc/Zc,  r^2 = x_n^2 + y_n^2
    x_d = x_n (1 + k1 r^2 + k2 r^4) + 2 p1 x_n y_n + p2 (r^2 + 2 x_n^2)
    y_d = y_n (1 + k1 r^2 + k2 r^4) + p1 (r^2 + 2 y_n^2) + 2 p2 x_n y_n
    u = f*mu*x_d + u0,   v = f*mv*y_d + v0

Run it: `py -m proxemix.salsa_camera` — self-test (camera placement, pixel ->
floor -> pixel round trip, the floor homography's agreement with the exact
model, and that image-down really is camera-ward).
"""
import os
import re

import numpy as np

DEFAULT_CALIB_DIR = os.path.join("data", "salsa", "calibration")

_KV = re.compile(r"^\s*([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$", re.M)


class SalsaCamera:
    def __init__(self, R, T, f, mu, mv, u0, v0, k1=0.0, k2=0.0, k3=0.0,
                 p1=0.0, p2=0.0, name="cam"):
        self.R = np.asarray(R, dtype=float).reshape(3, 3)   # camera -> world
        self.T = np.asarray(T, dtype=float).reshape(3)      # optical centre, world m
        self.f, self.mu, self.mv = float(f), float(mu), float(mv)
        self.u0, self.v0 = float(u0), float(v0)
        self.k1, self.k2, self.k3 = float(k1), float(k2), float(k3)
        self.p1, self.p2 = float(p1), float(p2)
        self.name = name

    # -- parsing ------------------------------------------------------------
    @classmethod
    def from_file(cls, path):
        txt = open(path, "r", encoding="utf-8", errors="replace").read()
        kv = dict(_KV.findall(txt))

        def g(k, d=0.0):
            return float(kv[k]) if k in kv else d

        R = [[g("R%d%d" % (i, j)) for j in (1, 2, 3)] for i in (1, 2, 3)]
        T = [g("T1"), g("T2"), g("T3")]
        return cls(R, T, g("f"), g("mu", 1.0), g("mv", 1.0), g("u0"), g("v0"),
                   g("k1"), g("k2"), g("k3"), g("p1"), g("p2"),
                   name=os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def cam(cls, n=1, calib_dir=DEFAULT_CALIB_DIR):
        return cls.from_file(os.path.join(calib_dir, "cam%d_calib.txt" % n))

    # -- forward ------------------------------------------------------------
    # SALSA's stored camera frame is x-left / y-up / z-forward, so the axes the
    # pixel model wants (x-right, y-down) are the stored ones negated — a 180
    # degree roll about the optical axis. Established by sweeping all eight
    # readings of (R or R', +/-x, +/-y) against the frames themselves: only
    # `R` with both signs flipped puts every annotated participant inside the
    # 1024x768 image AND projects a 1.7 m head *above* its floor point, with a
    # residual image tilt of 2.1 degrees (i.e. essentially no roll). `R'`
    # placed nobody in frame at all. See scratch/_probe_convention.py.
    _AXIS = np.array([-1.0, -1.0, 1.0])

    def world_to_cam(self, Xw):
        Xw = np.asarray(Xw, dtype=float).reshape(-1, 3)
        return ((Xw - self.T) @ self.R.T) * self._AXIS   # per row: diag(s) R (Xw - T)

    def _distort(self, xn, yn):
        r2 = xn * xn + yn * yn
        rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        xd = xn * rad + 2 * self.p1 * xn * yn + self.p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + self.p1 * (r2 + 2 * yn * yn) + 2 * self.p2 * xn * yn
        return xd, yd

    def project(self, Xw):
        """World metres -> pixels. Returns (N,2); rows behind the camera are NaN."""
        Xc = self.world_to_cam(Xw)
        z = Xc[:, 2]
        z = np.where(z > 1e-6, z, np.nan)
        xd, yd = self._distort(Xc[:, 0] / z, Xc[:, 1] / z)
        return np.column_stack([self.f * self.mu * xd + self.u0,
                                self.f * self.mv * yd + self.v0])

    def project_floor(self, xy, height_m=0.0):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return self.project(np.column_stack([xy, np.full(len(xy), float(height_m))]))

    # -- inverse ------------------------------------------------------------
    def _undistort_normalized(self, xd, yd, iters=20):
        """The same fixed-point iteration OpenCV's undistortPoints uses."""
        xn, yn = np.array(xd, dtype=float), np.array(yd, dtype=float)
        for _ in range(iters):
            r2 = xn * xn + yn * yn
            rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
            dx = 2 * self.p1 * xn * yn + self.p2 * (r2 + 2 * xn * xn)
            dy = self.p1 * (r2 + 2 * yn * yn) + 2 * self.p2 * xn * yn
            xn = (xd - dx) / rad
            yn = (yd - dy) / rad
        return xn, yn

    def pixel_ray(self, uv):
        """Pixels -> (origin, unit direction) in world metres."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        xd = (uv[:, 0] - self.u0) / (self.f * self.mu)
        yd = (uv[:, 1] - self.v0) / (self.f * self.mv)
        xn, yn = self._undistort_normalized(xd, yd)
        d_cam = np.column_stack([xn, yn, np.ones(len(xn))]) * self._AXIS
        d = d_cam @ self.R                        # per row: R' diag(s) d_cam
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        return self.T, d

    def to_floor(self, uv, height_m=0.0):
        """Pixels -> the plane z = height_m, in world metres. (N,2), NaN where
        the ray never meets the plane (points above the horizon)."""
        o, d = self.pixel_ray(uv)
        dz = d[:, 2]
        s = np.where(np.abs(dz) < 1e-9, np.nan, (height_m - o[2]) / dz)
        s = np.where(s > 0, s, np.nan)            # the plane is behind the lens
        return np.column_stack([o[0] + s * d[:, 0], o[1] + s * d[:, 1]])

    # -- the 3x3 the rest of Proxemix speaks --------------------------------
    def floor_homography(self, height_m=0.0, image_wh=(1024, 768)):
        """A cv2-style 3x3 mapping pixels -> floor metres, least-squares fitted
        over a grid in the lower image so `calibration.project_points` and
        `sam_video._world_dirs` can be used unchanged. It is only an
        approximation because of lens distortion; the self-test reports the
        residual, and nothing in the facing run relies on it."""
        import cv2
        w, h = image_wh
        gu, gv = np.meshgrid(np.linspace(0.08 * w, 0.92 * w, 9),
                             np.linspace(0.45 * h, 0.98 * h, 9))
        uv = np.column_stack([gu.ravel(), gv.ravel()])
        xy = self.to_floor(uv, height_m=height_m)
        ok = np.isfinite(xy).all(axis=1)
        H, _ = cv2.findHomography(uv[ok], xy[ok], method=0)
        return H

    def camera_ward(self, uv, height_m=0.0, delta_px=8.0):
        """Unit floor-plane vectors for "toward the camera" and "image right" at
        each pixel — the exact-model replacement for `sam_video._world_dirs`.

        `_world_dirs` takes floor-plane *image-down* as camera-ward, which is
        true only for a camera with no roll and no wide-angle distortion. On
        SALSA CAM1 the two differ by 4 to 25 degrees across the frame, so this
        returns the real thing: the direction from the floor point to the
        camera's own ground position. "Image right" stays a finite difference,
        because that is exactly what it means — the floor direction a person
        moves to appear further right in the picture.
        """
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        p0 = self.to_floor(uv, height_m)
        cam = self.T[:2][None, :] - p0
        right = self.to_floor(uv + np.array([delta_px, 0.0]), height_m) - p0
        out = []
        for v in (cam, right):
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n = np.where(n < 1e-9, np.nan, n)
            out.append(v / n)
        return out[0], out[1]


def self_test(calib_dir=DEFAULT_CALIB_DIR):
    c = SalsaCamera.cam(1, calib_dir)
    out = {"camera": c.name, "centre_m": [round(v, 3) for v in c.T]}

    # 1. the camera sits above the floor, near the annotated room's footprint
    assert 2.0 < c.T[2] < 5.0, "camera height %s m is not a mounted camera" % c.T[2]
    out["height_m"] = round(float(c.T[2]), 3)

    # 2. pixel -> floor -> pixel round trip
    uv = np.array([[200.0, 600.0], [512.0, 500.0], [850.0, 700.0], [512.0, 760.0]])
    xy = c.to_floor(uv)
    back = c.project_floor(xy)
    err = float(np.nanmax(np.abs(back - uv)))
    assert err < 1e-3, "round trip off by %s px" % err
    out["roundtrip_max_px"] = err
    out["probe_floor_m"] = [[round(v, 2) for v in r] for r in xy]

    # 3. the fitted floor homography against the exact model
    from proxemix.calibration import project_points
    H = c.floor_homography()
    approx = project_points(H, uv)
    out["homography_vs_exact_max_m"] = float(np.nanmax(np.abs(approx - xy)))

    # 4. a 1.7 m head projects ABOVE its floor point, near-vertically — the
    #    axis-convention check, in the form the frames themselves settled it
    floor = np.array([[9.0, 4.0], [11.0, 3.0], [7.0, 2.0]])
    feet = c.project_floor(floor)
    head = c.project(np.column_stack([floor, np.full(len(floor), 1.7)]))
    assert (head[:, 1] < feet[:, 1]).all(), "world-up does not project upward"
    tilt = np.degrees(np.arctan2(head[:, 0] - feet[:, 0], feet[:, 1] - head[:, 1]))
    out["standing_tilt_deg"] = [round(float(v), 1) for v in tilt]
    assert np.abs(tilt).max() < 30.0, "a standing person should be near-vertical"

    # 5. camera-ward is perpendicular-ish to image-right, and image-down is
    #    NOT camera-ward on this camera — which is why `_world_dirs` is not used
    probe = np.array([[400.0, 650.0], [512.0, 600.0], [800.0, 700.0]])
    camd, right = c.camera_ward(probe)
    out["camward_vs_imageright_deg"] = [
        round(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))), 1)
        for a, b in zip(camd, right)]
    imdown = c.to_floor(probe + np.array([0.0, 8.0])) - c.to_floor(probe)
    imdown = imdown / np.linalg.norm(imdown, axis=1, keepdims=True)
    out["imagedown_vs_camward_deg"] = [
        round(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))), 1)
        for a, b in zip(imdown, camd)]
    return out


if __name__ == "__main__":
    import json
    r = self_test()
    print(json.dumps(r, indent=2))
    print("\nsalsa_camera self-test PASSED — camera at %s m, "
          "pixel->floor->pixel within %.1e px." % (r["height_m"], r["roundtrip_max_px"]))
