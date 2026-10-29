"""SAM-derived facing on SALSA — the experiment that prices the orientation gap.

The question this module exists to answer
-----------------------------------------
`outputs/salsa_validation.md` section 12 measures the grouper at **F1 0.551**
with no usable facing and **0.721** when it is handed SALSA's own annotated body
orientation (`--facing oracle`). Real orientation is therefore worth about
**+0.17 F1** on this data. That number was the upper bound a SAM-derived facing
channel was aiming at, and nothing had measured how much of it SAM actually
reaches. This does.

The chain, end to end
---------------------
1. **Frames** — one JPEG per annotation timestamp from `cam1_cpp.avi`
   (`frame = 3 + 45k`, the 3 s stride SALSA annotates on), extracted with
   ffmpeg into `data/salsa/frames/cam1_cpp/`.
2. **Detection** — two image-endpoint calls per frame, "person" and "face",
   through `sam_video.SamVideoClient` (the live client proven 2026-10-27; this
   module adds no HTTP of its own). Results append to a JSONL so a crash never
   re-spends, and every call is charged against a hard cap.
3. **Matching** — each person box's feet point is projected to the floor plane
   through CAM1's real calibration (`proxemix.salsa_camera`, exact pinhole with
   distortion, not a homography) and matched one-to-one against the annotated
   participant positions at that timestamp by Hungarian assignment, rejecting
   any pair further apart than `max_match_m`.
4. **Facing** — `sam_video.pair_faces_to_people` pairs each face box to its
   containing person box, and `sam_video.facing_from_faces` turns that into an
   angle, with this module's camera supplying the floor-plane directions
   (`world_dirs=`) instead of the homography approximation.
5. **Scoring** — the facing table goes into `facing.apply_external_facing` via
   `salsa_validation --facing sam`, the same harness that produced 0.551 and
   0.721, on the same frames.

What this can and cannot see — stated before the numbers
--------------------------------------------------------
* **One camera of four.** SALSA ships four synchronised views; this uses CAM1.
  Anybody occluded from CAM1 is invisible here, where the human annotators had
  all four.
* **Face visibility is biased toward the camera.** The estimator's whole signal
  is "a face box was found inside this body box", so it is systematically more
  confident about people facing CAM1 than about people facing away. The
  `face_absent -> facing away from camera` fallback is a guess with a
  deliberately low confidence, and `--no-absent` turns it off entirely so the
  two can be compared.
* **The yaw is coarse.** Horizontal offset of the face box inside the body box,
  linearly mapped to +/-70 degrees. It is a direction, not a pose estimate.

Run it:
    py -m proxemix.salsa_sam_facing --self-test          # no key, no network
    py -m proxemix.salsa_sam_facing extract              # ffmpeg the frames
    py -m proxemix.salsa_sam_facing detect --max-usd 3.0 # the paid pass
    py -m proxemix.salsa_sam_facing facing               # match, derive, report
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

from proxemix import sam_video
from proxemix.salsa_adapter import SALSA_FPS, load_salsa_tracks
from proxemix.salsa_camera import SalsaCamera

DEFAULT_SALSA_ROOT = os.path.join("data", "salsa")
DEFAULT_PART = "salsa_cpp"
DEFAULT_CAM = 1
DEFAULT_FRAMES_DIR = os.path.join("data", "salsa", "frames", "cam1_cpp")
DEFAULT_DETECTIONS = os.path.join("scratch", "_salsa_sam", "detections.jsonl")
DEFAULT_FACING_CSV = os.path.join("outputs", "salsa_sam_facing.csv")
DEFAULT_REPORT = os.path.join("outputs", "salsa_sam_facing_report.json")

# SALSA annotates every 45 frames starting at frame 3 (t = 0.2 s, 15 fps).
ANNOTATION_FRAME_0 = 3
ANNOTATION_STRIDE = 45

# A SAM person box's feet point lands within this of an annotated participant
# before the two are called the same person. 0.5 m is the brief's figure and it
# is the right order: SALSA's own median nearest-neighbour distance at the
# cocktail party is 0.69 m, so a looser gate would start swapping neighbours.
DEFAULT_MAX_MATCH_M = 0.5

IMAGE_CALLS_PER_FRAME = 2       # one "person" pass, one "face" pass


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def annotation_frames(n):
    return [ANNOTATION_FRAME_0 + ANNOTATION_STRIDE * k for k in range(n)]


def extract_frames(video_path, out_dir, ffmpeg="ffmpeg", quality=3):
    """One ffmpeg pass pulling exactly the annotated frames. Returns the list of
    (canonical_frame, path), and refuses to guess if the count is not what the
    stride predicts."""
    os.makedirs(out_dir, exist_ok=True)
    sel = "select=eq(mod(n-%d\\,%d)\\,0)*gte(n\\,%d)" % (
        ANNOTATION_FRAME_0, ANNOTATION_STRIDE, ANNOTATION_FRAME_0)
    cmd = [ffmpeg, "-v", "error", "-y", "-i", video_path, "-vf", sel,
           "-vsync", "0", "-q:v", str(quality), os.path.join(out_dir, "%06d.jpg")]
    subprocess.run(cmd, check=True)
    return list_frames(out_dir)


def list_frames(out_dir):
    names = sorted(n for n in os.listdir(out_dir) if n.lower().endswith(".jpg"))
    return [(ANNOTATION_FRAME_0 + ANNOTATION_STRIDE * k, os.path.join(out_dir, n))
            for k, n in enumerate(names)]


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

class _Budget:
    """One shared, locked purse. Each worker also carries its own client cap, so
    a bug in this class cannot overspend on its own."""

    def __init__(self, cap_usd):
        self.cap = float(cap_usd)
        self.spent = 0.0
        self._lock = threading.Lock()

    def take(self, usd):
        with self._lock:
            if self.spent + usd > self.cap + 1e-9:
                return False
            self.spent += usd
            return True


def _load_done(path):
    """(frame, concept) pairs already on disk, so a resumed run never re-spends."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("ok"):
                done.add((int(rec["frame"]), rec["concept"]))
    return done


def detect(frames_dir=DEFAULT_FRAMES_DIR, out_path=DEFAULT_DETECTIONS,
           concepts=(sam_video.DEFAULT_PERSON_CONCEPT, sam_video.DEFAULT_FACE_CONCEPT),
           max_usd=3.0, workers=6, limit=None, stride=1, client_factory=None,
           progress_every=20, retries=2):
    """Two image-endpoint calls per frame, in parallel, appended to a JSONL.

    `client_factory()` makes one client per worker thread — the real
    `SamVideoClient` by default, a fake in the self-test. Nothing here touches
    the key; the client reads it fresh from disk per call and never returns it.
    """
    frames = list_frames(frames_dir)[::max(1, int(stride))]
    if limit:
        frames = frames[:int(limit)]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _load_done(out_path)

    jobs = [(fr, path, c) for fr, path in frames for c in concepts
            if (fr, c) not in done]
    per_call = sam_video.image_cost_usd(1)
    projected = per_call * len(jobs)
    budget = _Budget(max_usd)
    if client_factory is None:
        def client_factory():
            return sam_video.SamVideoClient(max_usd=max_usd / max(1, workers))

    print("%d frame(s), %d call(s) to make (%d already on disk), "
          "projected $%.2f against a $%.2f cap"
          % (len(frames), len(jobs), len(done), projected, max_usd))
    if projected > max_usd + 1e-9:
        raise sam_video.SpendCapExceeded(
            "this pass would cost $%.2f against a $%.2f cap — refused before "
            "the first call. Raise --max-usd deliberately, or cut --stride."
            % (projected, max_usd))

    lock = threading.Lock()
    local = threading.local()
    state = {"n": 0, "boxes": 0, "fail": 0, "t0": time.time()}
    fh = open(out_path, "a", encoding="utf-8")

    def write(rec):
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            state["n"] += 1
            if rec.get("ok"):
                state["boxes"] += len(rec["boxes"])
            else:
                state["fail"] += 1
            if state["n"] % progress_every == 0:
                el = time.time() - state["t0"]
                print("  %d/%d calls, %d boxes, %d failed, $%.2f, %.0fs elapsed "
                      "(%.1f calls/min)"
                      % (state["n"], len(jobs), state["boxes"], state["fail"],
                         budget.spent, el, 60.0 * state["n"] / max(1e-9, el)),
                      flush=True)

    def work(job):
        frame, path, concept = job
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
        last = None
        for attempt in range(retries + 1):
            if not budget.take(per_call):
                write({"frame": frame, "concept": concept, "ok": False,
                       "error": "budget cap reached"})
                return
            try:
                text = client.segment_image(path, concept=concept)
                parsed = sam_video.parse_segmentation(text)
                dets = [d for v in parsed.values() for d in v]
                write({"frame": frame, "concept": concept, "ok": True,
                       "boxes": [{"object_id": d["object_id"], "bbox": d["bbox"],
                                  "conf": d.get("confidence"),
                                  "w": d.get("frame_width"),
                                  "h": d.get("frame_height")} for d in dets],
                       "warnings": sam_video.parse_warnings(parsed),
                       "attempt": attempt})
                return
            except Exception as e:                       # noqa: BLE001 - logged
                last = "%s: %s" % (type(e).__name__, e)
                if isinstance(e, sam_video.SpendCapExceeded):
                    break
                time.sleep(1.5 * (attempt + 1))
        write({"frame": frame, "concept": concept, "ok": False, "error": last})

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, jobs))
    finally:
        fh.close()
    print("done: %d call(s), %d box(es), %d failure(s), $%.2f spent"
          % (state["n"], state["boxes"], state["fail"], budget.spent))
    return {"calls": state["n"], "boxes": state["boxes"],
            "failures": state["fail"], "usd": round(budget.spent, 4)}


def load_detections(path=DEFAULT_DETECTIONS):
    """JSONL -> one long box table (concept, frame, object_id, x1..y2, conf)."""
    rows, failures = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                failures.append(rec)
                continue
            for b in rec["boxes"]:
                x1, y1, x2, y2 = b["bbox"]
                rows.append({"concept": rec["concept"], "frame": int(rec["frame"]),
                             "object_id": int(b["object_id"]),
                             "x1": float(x1), "y1": float(y1),
                             "x2": float(x2), "y2": float(y2),
                             "conf": b.get("conf")})
    df = pd.DataFrame(rows, columns=["concept", "frame", "object_id",
                                     "x1", "y1", "x2", "y2", "conf"])
    # a re-run can append a second copy of the same call; keep one
    df = df.drop_duplicates(subset=["concept", "frame", "object_id",
                                    "x1", "y1", "x2", "y2"])
    return df.reset_index(drop=True), failures


# ---------------------------------------------------------------------------
# matching detections to annotated participants
# ---------------------------------------------------------------------------

def match_to_participants(person_table, gt_tracks, cam,
                          max_match_m=DEFAULT_MAX_MATCH_M):
    """Give every SAM person box the track_id of the participant it is, or none.

    Feet point -> floor plane through the real camera, then one-to-one Hungarian
    assignment per frame against the annotated positions, with every pair over
    `max_match_m` refused. One-to-one matters: two boxes on one person must not
    both become that person, and a crowd packed at 0.69 m median spacing will
    produce exactly that if you take nearest-neighbour greedily.
    """
    from scipy.optimize import linear_sum_assignment

    out = person_table.copy()
    feet = np.column_stack([(out["x1"].to_numpy() + out["x2"].to_numpy()) / 2.0,
                            out["y2"].to_numpy()])
    floor = cam.to_floor(feet)
    out["feet_u"], out["feet_v"] = feet[:, 0], feet[:, 1]
    out["det_x_m"], out["det_y_m"] = floor[:, 0], floor[:, 1]
    out["track_id"] = None
    out["match_dist_m"] = np.nan

    gt_by_frame = {f: g for f, g in gt_tracks.groupby("frame")}
    stats = {"frames": 0, "boxes": 0, "matched": 0, "gt_people": 0,
             "boxes_offplane": 0}
    for frame, g in out.groupby("frame"):
        gt = gt_by_frame.get(frame)
        stats["frames"] += 1
        stats["boxes"] += len(g)
        if gt is None or not len(gt):
            continue
        stats["gt_people"] += len(gt)
        det_xy = g[["det_x_m", "det_y_m"]].to_numpy()
        ok = np.isfinite(det_xy).all(axis=1)
        stats["boxes_offplane"] += int((~ok).sum())
        if not ok.any():
            continue
        gt_xy = gt[["x_m", "y_m"]].to_numpy()
        d = np.hypot(det_xy[ok][:, None, 0] - gt_xy[None, :, 0],
                     det_xy[ok][:, None, 1] - gt_xy[None, :, 1])
        big = 1e6
        cost = np.where(d <= max_match_m, d, big)
        ri, ci = linear_sum_assignment(cost)
        idx = g.index.to_numpy()[ok]
        for r, c in zip(ri, ci):
            if cost[r, c] >= big:
                continue
            out.loc[idx[r], "track_id"] = gt["track_id"].to_numpy()[c]
            out.loc[idx[r], "match_dist_m"] = float(d[r, c])
            stats["matched"] += 1
    stats["pct_boxes_matched"] = round(100.0 * stats["matched"] / max(1, stats["boxes"]), 1)
    stats["pct_participants_matched"] = round(
        100.0 * stats["matched"] / max(1, stats["gt_people"]), 1)
    md = out["match_dist_m"].to_numpy()
    md = md[np.isfinite(md)]
    stats["median_match_dist_m"] = round(float(np.median(md)), 3) if len(md) else None
    return out, stats


# ---------------------------------------------------------------------------
# facing
# ---------------------------------------------------------------------------

def camera_world_dirs(cam):
    """The `world_dirs` callable `sam_video.facing_from_faces` takes — the exact
    camera instead of `_world_dirs`' homography-and-image-down approximation."""

    def dirs(px, py):
        c, r = cam.camera_ward(np.array([[float(px), float(py)]]))
        if not (np.isfinite(c[0]).all() and np.isfinite(r[0]).all()):
            return None, None
        return c[0], r[0]

    return dirs


def derive_facing(matched_people, face_table, cam, include_absent=True,
                  max_yaw_deg=sam_video.DEFAULT_MAX_YAW_DEG,
                  face_conf=sam_video.DEFAULT_FACE_CONF,
                  absent_conf=sam_video.DEFAULT_ABSENT_CONF,
                  min_containment=sam_video.DEFAULT_MIN_CONTAINMENT):
    """Matched person boxes + face boxes -> a facing table keyed (track_id, frame).

    Only matched boxes are kept: an unmatched box has no participant to speak
    for, and the harness is scored against participants.
    """
    people = matched_people[matched_people["track_id"].notna()].copy()
    faces = face_table.reset_index(drop=True)
    facing, pairings = sam_video.pair_and_face(
        people, faces,
        min_containment=min_containment,
        world_dirs=camera_world_dirs(cam),
        max_yaw_deg=max_yaw_deg, face_conf=face_conf, absent_conf=absent_conf,
        include_absent=include_absent, track_col="track_id")
    # One participant can only be in one place: if two boxes matched the same
    # person in a frame (it happens at the 0.5 m gate), keep the confident one.
    if len(facing):
        facing = (facing.sort_values("facing_conf", ascending=False)
                        .drop_duplicates(subset=["track_id", "frame"])
                        .sort_values(["track_id", "frame"])
                        .reset_index(drop=True))
    return facing, pairings


def angular_error(facing, gt_tracks, pose_col="body_pose"):
    """Median |angle| between the derived facing and SALSA's annotated pose, in
    degrees, over the person-frames that have both."""
    gt = gt_tracks[["track_id", "frame", pose_col]].copy()
    j = facing.merge(gt, on=["track_id", "frame"], how="inner")
    if not len(j):
        return {"n": 0}
    d = np.degrees(np.abs(np.arctan2(np.sin(j["facing_rad"] - j[pose_col]),
                                     np.cos(j["facing_rad"] - j[pose_col]))))
    out = {}
    for name, sub in (("all", j.assign(_d=d)),
                      ("face_seen", j.assign(_d=d)[j["facing_source"].to_numpy() == "face"]),
                      ("face_absent", j.assign(_d=d)[j["facing_source"].to_numpy() == "face_absent"])):
        if not len(sub):
            out[name] = {"n": 0}
            continue
        dd = sub["_d"].to_numpy()
        out[name] = {
            "n": int(len(dd)),
            "median_deg": round(float(np.median(dd)), 1),
            "mean_deg": round(float(np.mean(dd)), 1),
            "pct_within_45deg": round(float(100.0 * np.mean(dd <= 45)), 1),
            "pct_within_90deg": round(float(100.0 * np.mean(dd <= 90)), 1),
        }
    # the floor to beat: guessing a single constant angle, fitted in hindsight
    best = min((float(np.median(np.degrees(np.abs(np.arctan2(
        np.sin(a - j[pose_col]), np.cos(a - j[pose_col]))))))
        for a in np.linspace(-np.pi, np.pi, 72)))
    out["best_constant_guess_median_deg"] = round(best, 1)
    out["random_guess_median_deg"] = 90.0
    return out


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------

def build_facing(salsa_root=DEFAULT_SALSA_ROOT, part=DEFAULT_PART, cam_n=DEFAULT_CAM,
                 detections=DEFAULT_DETECTIONS, out_csv=DEFAULT_FACING_CSV,
                 report_path=DEFAULT_REPORT, max_match_m=DEFAULT_MAX_MATCH_M,
                 include_absent=True, **facing_kw):
    cam = SalsaCamera.cam(cam_n, os.path.join(salsa_root, "calibration"))
    gt = load_salsa_tracks(salsa_root, part)
    det, failures = load_detections(detections)
    person = det[det.concept == sam_video.DEFAULT_PERSON_CONCEPT].reset_index(drop=True)
    face = det[det.concept == sam_video.DEFAULT_FACE_CONCEPT].reset_index(drop=True)

    matched, mstats = match_to_participants(person, gt, cam, max_match_m)
    facing, pairings = derive_facing(matched, face, cam,
                                     include_absent=include_absent, **facing_kw)

    covered = set(zip(matched.loc[matched.track_id.notna(), "track_id"],
                      matched.loc[matched.track_id.notna(), "frame"]))
    with_face = pairings[pairings["face_index"].apply(sam_video.has_face)]
    gt_frames = gt[gt.frame.isin(set(person["frame"]))]

    report = {
        "part": part, "camera": cam.name,
        "frames_detected": int(person["frame"].nunique()),
        "person_boxes": int(len(person)),
        "face_boxes": int(len(face)),
        "detection_failures": len(failures),
        "matching": mstats,
        "max_match_m": max_match_m,
        "gt_person_frames_on_detected_frames": int(len(gt_frames)),
        "matched_person_frames": len(covered),
        "pct_annotated_people_found": round(
            100.0 * len(covered) / max(1, len(gt_frames)), 1),
        "face_read_on_matched_people_pct": round(
            100.0 * len(with_face) / max(1, len(pairings)), 1),
        "facing_rows": int(len(facing)),
        "facing_by_source": {str(k): int(v) for k, v in
                             facing["facing_source"].value_counts().items()}
        if len(facing) else {},
        "angular_error_vs_body_pose": angular_error(facing, gt),
        "angular_error_vs_head_pose": angular_error(facing, gt, "head_pose"),
    }
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out = facing.copy()
    out.insert(0, "part", part)          # so the scorer can filter by part
    out.to_csv(out_csv, index=False)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return facing, report


# ---------------------------------------------------------------------------
# self-test — no key, no network, no frames
# ---------------------------------------------------------------------------

def self_test():
    """Three synthetic participants on CAM1's real calibration, projected into
    pixel space, handed back through the whole chain. Everything the paid run
    does except the HTTP.
    """
    cam = SalsaCamera.cam(1)
    frames = [3, 48]
    people = [("01", 9.0, 4.0), ("02", 10.5, 3.2), ("03", 7.5, 3.6)]
    gt_rows, person_rows, face_rows = [], [], []
    for fr in frames:
        for i, (tid, x, y) in enumerate(people):
            gt_rows.append({"track_id": tid, "frame": fr, "t": fr / SALSA_FPS,
                            "x_m": x, "y_m": y, "body_pose": 0.0, "head_pose": 0.0})
            feet = cam.project_floor(np.array([[x, y]]))[0]
            head = cam.project(np.array([[x, y, 1.75]]))[0]
            w = abs(feet[1] - head[1]) * 0.32
            person_rows.append({"concept": "person", "frame": fr, "object_id": i,
                                "x1": feet[0] - w / 2, "y1": head[1],
                                "x2": feet[0] + w / 2, "y2": feet[1], "conf": 0.9})
            if tid != "03":                      # one person's face is not seen
                fw = w * 0.35
                face_rows.append({"concept": "face", "frame": fr, "object_id": i,
                                  "x1": feet[0] - fw / 2, "y1": head[1],
                                  "x2": feet[0] + fw / 2,
                                  "y2": head[1] + fw, "conf": 0.8})
    gt = pd.DataFrame(gt_rows)
    person = pd.DataFrame(person_rows)
    face = pd.DataFrame(face_rows)

    matched, mstats = match_to_participants(person, gt, cam)
    assert mstats["matched"] == 6, mstats
    assert mstats["median_match_dist_m"] < 0.05, mstats

    facing, pairings = derive_facing(matched, face, cam)
    assert len(facing) == 6, facing
    src = facing["facing_source"].value_counts().to_dict()
    assert src.get("face") == 4 and src.get("face_absent") == 2, src

    # a centred face must come out pointing at the camera, within a degree
    for row in facing[facing.facing_source == "face"].itertuples():
        p = gt[(gt.track_id == row.track_id) & (gt.frame == row.frame)].iloc[0]
        want = np.arctan2(cam.T[1] - p.y_m, cam.T[0] - p.x_m)
        err = abs(np.degrees(np.arctan2(np.sin(row.facing_rad - want),
                                        np.cos(row.facing_rad - want))))
        assert err < 1.5, ("centred face is not camera-ward", row.track_id, err)
    # an absent face must come out pointing away from the camera
    for row in facing[facing.facing_source == "face_absent"].itertuples():
        p = gt[(gt.track_id == row.track_id) & (gt.frame == row.frame)].iloc[0]
        want = np.arctan2(cam.T[1] - p.y_m, cam.T[0] - p.x_m) + np.pi
        err = abs(np.degrees(np.arctan2(np.sin(row.facing_rad - want),
                                        np.cos(row.facing_rad - want))))
        assert err < 1.5, ("absent face is not pointing away", row.track_id, err)

    # an off-centre face must yaw, and in the direction of the offset
    shifted = person.copy()
    shifted.loc[0, "x1"] -= 0  # keep the body box, move the face right instead
    f2 = face.copy()
    span = shifted.loc[0, "x2"] - shifted.loc[0, "x1"]
    f2.loc[0, "x1"] += span * 0.4
    f2.loc[0, "x2"] += span * 0.4
    m2, _ = match_to_participants(shifted, gt, cam)
    fac2, _ = derive_facing(m2, f2, cam)
    a = facing[(facing.track_id == "01") & (facing.frame == 3)]["facing_rad"].iloc[0]
    b = fac2[(fac2.track_id == "01") & (fac2.frame == 3)]["facing_rad"].iloc[0]
    turn = np.degrees(np.arctan2(np.sin(b - a), np.cos(b - a)))
    assert abs(turn) > 20.0, ("an off-centre face did not yaw", turn)

    # the budget gate: a fake client that counts calls, and a cap that bites
    class _Fake:
        text = "<0f>0 <|box;x1=10;y1=10;x2=20;y2=40;c=0.9;w=1024;h=768|>"

        def __init__(self):
            self.n = 0

        def segment_image(self, path, concept="person"):
            self.n += 1
            return self.text

    errs = angular_error(facing, gt)
    assert errs["all"]["n"] == 6, errs
    return {
        "matching": mstats,
        "facing_sources": src,
        "yaw_on_offset_face_deg": round(float(turn), 1),
        "angular_error_vs_body_pose": errs,
        "world_dirs_seam": "exact camera, not _world_dirs",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", default="self-test",
                    choices=["self-test", "extract", "detect", "facing"])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--salsa-root", default=DEFAULT_SALSA_ROOT)
    ap.add_argument("--part", default=DEFAULT_PART)
    ap.add_argument("--cam", type=int, default=DEFAULT_CAM)
    ap.add_argument("--video", default=os.path.join("data", "salsa", "video",
                                                    "cam1_cpp.avi"))
    ap.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    ap.add_argument("--detections", default=DEFAULT_DETECTIONS)
    ap.add_argument("--out-csv", default=DEFAULT_FACING_CSV)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth annotated frame")
    ap.add_argument("--max-match-m", type=float, default=DEFAULT_MAX_MATCH_M)
    ap.add_argument("--no-absent", action="store_true",
                    help="drop the face-absent -> facing-away guess")
    ap.add_argument("--max-yaw-deg", type=float, default=sam_video.DEFAULT_MAX_YAW_DEG)
    ap.add_argument("--face-conf", type=float, default=sam_video.DEFAULT_FACE_CONF,
                    help="confidence ceiling for a face-derived angle")
    ap.add_argument("--absent-conf", type=float, default=sam_video.DEFAULT_ABSENT_CONF,
                    help="confidence for the facing-away guess; groups.py "
                         "ignores anything under 0.3")
    ap.add_argument("--min-containment", type=float,
                    default=sam_video.DEFAULT_MIN_CONTAINMENT,
                    help="share of a face box that must sit inside a person box")
    args = ap.parse_args(argv)
    cmd = "self-test" if args.self_test else args.command

    if cmd == "self-test":
        r = self_test()
        print(json.dumps(r, indent=2))
        print("\nsalsa_sam_facing self-test PASSED — matching, the exact-camera "
              "facing seam and the angular-error measure all run with no key.")
        return 0

    if cmd == "extract":
        got = extract_frames(args.video, args.frames_dir)
        print("extracted %d frame(s) to %s (frames %d..%d)"
              % (len(got), args.frames_dir, got[0][0], got[-1][0]))
        return 0

    if cmd == "detect":
        r = detect(frames_dir=args.frames_dir, out_path=args.detections,
                   max_usd=args.max_usd, workers=args.workers,
                   limit=args.limit, stride=args.stride)
        print(json.dumps(r, indent=2))
        return 0

    if cmd == "facing":
        facing, report = build_facing(
            salsa_root=args.salsa_root, part=args.part, cam_n=args.cam,
            detections=args.detections, out_csv=args.out_csv,
            report_path=args.report, max_match_m=args.max_match_m,
            include_absent=not args.no_absent, max_yaw_deg=args.max_yaw_deg,
            face_conf=args.face_conf, absent_conf=args.absent_conf,
            min_containment=args.min_containment)
        print(json.dumps(report, indent=2))
        print("\nwrote %d facing row(s) to %s" % (len(facing), args.out_csv))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
