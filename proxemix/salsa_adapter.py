"""SALSA adapter — loads the SALSA dataset's human position annotations into
the canonical track table (track_id, frame, x, y) that every later Proxemix
stage expects, plus its F-formation (conversation-group) ground truth.

Same job as `grand_central_adapter.py`, different source dataset. SALSA is
used here because, unlike Grand Central, it carries *human-annotated
conversation groups* — which is what makes it a validation set for
`proxemix/groups.py` rather than just more input data.

Source
------
SALSA, Technologies of Vision lab, Fondazione Bruno Kessler.
<https://tev.fbk.eu/resources/salsa> — CC BY-NC-SA 3.0. Provenance, licence
and download date are recorded in `data/salsa/SOURCE.md`.

Cite: Alameda-Pineda, Staiano, Subramanian, Batrinca, Ricci, Lepri, Lanz,
Sebe. "SALSA: A Novel Dataset for Multimodal Group Behavior Analysis."
IEEE TPAMI, 2015.

Source format
-------------
Two parts, each a separate ~25-32 minute recording of the same 18 people:
`salsa_ps` (poster session) and `salsa_cpp` (cocktail party).

`Annotation/<part>/geometryGT/NN.csv` — one file per participant, NN = 01..18,
comma-separated, no header. Column meaning is quoted verbatim from the
dataset's own `README-SALSA.txt`, section 3.1:

    Timestamp[s]  Ground_Position_X[m]  Ground_Position_Y[m]  Useless_Field
    Body_Pose[rad]  Relative_Head2Body_Pose[rad]  Validity[bool]

`Annotation/<part>/fformationGT.csv` — the conversation-group ground truth.
Verbatim from README section 3.2:

    Timestamp[s] ID OF THE GROUP MEMBERS SEPARATED BY SPACES

    "Therefore, the rows vary in length. In addition, there are many rows per
    timestamps, since there are many groups at every time."

COORDINATE FRAME — established, not assumed
-------------------------------------------
**SALSA positions are already on the floor plane in metres, in a single world
frame shared by all participants and both parts. No homography or unit
conversion is applied by this adapter.**

Three independent pieces of evidence, in descending order of authority:

1. *The dataset says so.* `README-SALSA.txt` names the columns
   `Ground_Position_X[m]` and `Ground_Position_Y[m]` — "Ground_Position"
   (floor plane, not image plane) and an explicit `[m]` unit. The dataset
   also ships separate per-camera calibration matrices for projecting *pixels*
   to this plane, which only makes sense if the geometry ground truth is
   already past that step.

2. *The room is room-sized.* Measured off the files themselves (see
   `sanity_check()` below, run 2026-10-25):
     - poster session: x spans 8.39-14.87, y spans 0.73-6.79 -> a 6.5 m x 6.1 m area
     - cocktail party: x spans 3.91-14.82, y spans 0.08-6.66 -> a 10.9 m x 6.6 m area
   Both are plausible indoor spaces for 18 people. Under a centimetre reading
   the venue would be 6 cm across; under a pixel reading, 6 pixels. Only metres
   survive.

3. *Inter-person distances fall where human proxemics says they should.* The
   brief's own check — "inter-person distances in a cocktail party should mostly
   fall 0.3-4 m". Measured over every annotated timestamp, all valid pairs:
     - poster session: median pair distance 2.52 m; 81.7% of pairs in 0.30-4.00 m;
       0.03% under 0.30 m; 0% over 10 m. Nearest-neighbour distance median 0.60 m,
       with 99.5% of nearest-neighbour distances in 0.30-4.00 m.
     - cocktail party: median pair distance 2.32 m; 84.9% in 0.30-4.00 m;
       0.06% under 0.30 m; 0% over 10 m. Nearest-neighbour median 0.69 m, 99.0%
       in range.
   Nobody ever occupies the same half-metre as someone else, and nobody is ever
   ten metres away. That is exactly the signature of 18 people in one room,
   measured in metres.

The single caveat worth stating aloud: the two parts do **not** share an origin
in any verified way — the cocktail-party x-range starts 4.5 m further left, so
the recordings are plausibly different areas of the venue (or the same area with
a different annotated extent). This adapter therefore never mixes the two parts
into one table; each part is loaded, grouped and scored separately.

Frames and time
---------------
SALSA video is 15 fps and "Position, head and body orientation of each target
was annotated every 45 frames (3 seconds)" (project page). Timestamps in the
files run 0.2, 3.2, 6.2, ... confirming the 3 s stride. This module therefore
sets `SALSA_FPS = 15.0` and derives `frame = round(t * 15)`, giving 3, 48, 93,
... — 45-frame steps, matching the dataset's own description.

Note `proxemix/timecode.py` hard-codes `FPS = 25.0` for the Grand Central data.
That module is not edited here; anything time-formatted for SALSA must use
`SALSA_FPS`. The canonical table this adapter returns already carries a `t`
column in seconds, read straight from the annotation files, so nothing
downstream needs to re-derive it.

Validity
--------
Rows whose `Validity[bool]` is 0 are dropped — the annotation itself declares
them unreliable. Measured: 56 of 11,610 rows (0.48%) in the poster session,
351 of 9,000 rows (3.90%) in the cocktail party.
"""
import glob
import os

import numpy as np
import pandas as pd

SALSA_FPS = 15.0
SALSA_PARTS = ("salsa_ps", "salsa_cpp")
PART_LABELS = {"salsa_ps": "poster session", "salsa_cpp": "cocktail party"}

GEOMETRY_COLUMNS = [
    "t",             # Timestamp[s]
    "x",             # Ground_Position_X[m]
    "y",             # Ground_Position_Y[m]
    "_unused",       # "Useless_Field", per the dataset's own README
    "body_pose",     # Body_Pose[rad]
    "head_rel_pose", # Relative_Head2Body_Pose[rad]
    "valid",         # Validity[bool]
]


def part_dir(salsa_root, part):
    """Resolve <salsa_root>/Annotation/<part>, which is where both tgz
    archives extract to."""
    return os.path.join(salsa_root, "Annotation", part)


def load_salsa_tracks(salsa_root, part, drop_invalid=True):
    """Load one SALSA part's position annotations into the canonical track
    table.

    Returns a DataFrame with columns:
        track_id  str, zero-padded participant number ("01".."18")
        frame     int, derived as round(t * SALSA_FPS)
        x, y      float, floor-plane metres (identical to x_m/y_m — SALSA
                  positions are already calibrated; see the module docstring)
        t         float, seconds from the first video frame
        x_m, y_m  float, the metre columns the metrics modules read
        body_pose, head_pose  float, radians (head_pose = body + relative,
                  per README section 3.1). Carried through because they are
                  real annotations, though nothing in the keyless Proxemix
                  core consumes orientation yet.
        valid     int

    Sorted by track_id then frame, matching `load_grand_central_tracks`.
    """
    gdir = os.path.join(part_dir(salsa_root, part), "geometryGT")
    files = sorted(glob.glob(os.path.join(gdir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"no geometryGT .csv files found under {gdir} — has "
            f"SALSA_Annotation_{'ps' if part == 'salsa_ps' else 'cpp'}.tgz been extracted? "
            f"See data/salsa/SOURCE.md."
        )

    per_person = []
    for fp in files:
        track_id = os.path.splitext(os.path.basename(fp))[0]
        vals = np.loadtxt(fp, delimiter=",")
        if vals.size == 0:
            continue
        vals = np.atleast_2d(vals)
        if vals.shape[1] != len(GEOMETRY_COLUMNS):
            raise ValueError(
                f"{fp}: expected {len(GEOMETRY_COLUMNS)} columns per README section 3.1, "
                f"got {vals.shape[1]}"
            )
        d = pd.DataFrame(vals, columns=GEOMETRY_COLUMNS)
        d["track_id"] = track_id
        per_person.append(d)

    df = pd.concat(per_person, ignore_index=True)
    if drop_invalid:
        df = df[df["valid"] == 1].copy()

    df["frame"] = np.rint(df["t"] * SALSA_FPS).astype(int)
    df["x_m"] = df["x"]
    df["y_m"] = df["y"]
    df["head_pose"] = df["body_pose"] + df["head_rel_pose"]
    df["valid"] = df["valid"].astype(int)

    out = df[[
        "track_id", "frame", "x", "y", "t", "x_m", "y_m",
        "body_pose", "head_pose", "valid",
    ]]
    return out.sort_values(["track_id", "frame"]).reset_index(drop=True)


def load_salsa_fformations(salsa_root, part, min_cardinality=2):
    """Load one part's F-formation ground truth.

    Returns a DataFrame: t, frame, group_key, n_members — one row per
    annotated group per timestamp, where group_key is the comma-joined sorted
    member track_ids, matching the `group_key` convention `proxemix.groups`
    emits so the two are directly comparable.

    `min_cardinality=2` drops the singleton rows SALSA includes (a row naming
    one person). That follows the F-formation literature, which defines an
    F-formation as involving two or more people (Kendon) and evaluates
    cardinalities k=2..6 only; Setti et al. call a lone individual a "spurious
    F-formation" kept "for the sake of mathematical simplicity" in the model,
    not a group to be scored. Pass `min_cardinality=1` to include them as a
    sensitivity check.

    Exact duplicate rows (the same member set listed twice at the same
    timestamp — SALSA's poster-session file does this, e.g. participant 18
    appears as a singleton four times at t=0.2) are collapsed.
    """
    fp = os.path.join(part_dir(salsa_root, part), "fformationGT.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"missing F-formation ground truth: {fp}")

    rows = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = [tok.strip() for tok in line.split(",")]
            t = float(fields[0])
            members = sorted(f"{int(tok):02d}" for tok in fields[1:] if tok)
            if len(members) < min_cardinality:
                continue
            rows.append({
                "t": t,
                "frame": int(round(t * SALSA_FPS)),
                "group_key": ",".join(members),
                "n_members": len(members),
            })

    df = pd.DataFrame(rows, columns=["t", "frame", "group_key", "n_members"])
    df = df.drop_duplicates(subset=["t", "group_key"]).reset_index(drop=True)
    return df.sort_values(["t", "group_key"]).reset_index(drop=True)


def sanity_check(salsa_root, parts=SALSA_PARTS):
    """Re-run the coordinate-frame evidence quoted in the module docstring, so
    the claim "these are metres on the floor plane" is checkable rather than
    asserted. Returns a list of dicts, one per part.

    Run it: `py -m proxemix.salsa_adapter`
    """
    results = []
    for part in parts:
        tracks = load_salsa_tracks(salsa_root, part)
        pair_d, nn_d = [], []
        for _, g in tracks.groupby("frame"):
            xy = g[["x_m", "y_m"]].to_numpy()
            if len(xy) < 2:
                continue
            d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
            pair_d.append(d[np.triu_indices(len(xy), 1)])
            np.fill_diagonal(d, np.inf)
            nn_d.append(d.min(axis=1))
        pair_d = np.concatenate(pair_d)
        nn_d = np.concatenate(nn_d)

        ff = load_salsa_fformations(salsa_root, part)
        results.append({
            "part": part,
            "label": PART_LABELS[part],
            "n_people": int(tracks["track_id"].nunique()),
            "n_rows": int(len(tracks)),
            "n_timestamps": int(tracks["frame"].nunique()),
            "duration_s": float(tracks["t"].max() - tracks["t"].min()),
            "extent_x_m": (float(tracks["x_m"].min()), float(tracks["x_m"].max())),
            "extent_y_m": (float(tracks["y_m"].min()), float(tracks["y_m"].max())),
            "pair_dist_median_m": float(np.median(pair_d)),
            "pair_dist_pct_in_0p3_to_4m": float(100 * np.mean((pair_d >= 0.3) & (pair_d <= 4.0))),
            "pair_dist_pct_under_0p3m": float(100 * np.mean(pair_d < 0.3)),
            "pair_dist_pct_over_10m": float(100 * np.mean(pair_d > 10.0)),
            "nn_dist_median_m": float(np.median(nn_d)),
            "nn_dist_pct_in_0p3_to_4m": float(100 * np.mean((nn_d >= 0.3) & (nn_d <= 4.0))),
            "gt_groups_rows": int(len(ff)),
            "gt_cardinality_counts": {int(k): int(v) for k, v in
                                      ff["n_members"].value_counts().sort_index().items()},
        })
    return results


if __name__ == "__main__":
    import json
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "salsa")
    for r in sanity_check(root):
        print(json.dumps(r, indent=2))
        ex, ey = r["extent_x_m"], r["extent_y_m"]
        print(f"  -> {r['label']}: {ex[1] - ex[0]:.1f} m x {ey[1] - ey[0]:.1f} m room, "
              f"{r['n_people']} people, {r['duration_s'] / 60:.1f} min, "
              f"{r['pair_dist_pct_in_0p3_to_4m']:.1f}% of pairs within 0.3-4 m "
              f"-> consistent with floor-plane metres.\n")
