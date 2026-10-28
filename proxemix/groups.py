"""Groups: conversation-group (F-formation) detection per frame, then temporal
persistence.

WHAT CHANGED, AND WHY (2026-10-26)
----------------------------------
This module used to be plain DBSCAN on position. `outputs/salsa_validation.md`
measured that against SALSA's 18-person human F-formation annotations and found
two defects and one ceiling:

* **The 2.0 m default radius chained a dense room into one blob** — 427 of the
  poster session's 811 detections were a single cluster containing all 18
  people, F1 0.069 out of the box.
* **`debounce_frames=40` was silently coupled to Grand Central's sampling
  stride.** A dataset sampled more sparsely (SALSA: every 45 frames) produced
  zero group episodes with no error raised.
* **DBSCAN is single-linkage, so it cannot refuse a chain.** A joins B if they
  are within eps, and the cluster then extends through anyone within eps of B,
  and onward. 2,353 of the tuned setting's misses were ground-truth groups
  swallowed into a larger detection.

All three are fixed here. The linking radius default is 1.1 m and is no longer
the thing holding chains back, the persistence filter counts **seconds** and
adapts to the data's own sampling stride, and the clusterer is no longer
single-linkage. Measured effect on SALSA, same harness, same criterion:
**F1 0.385 -> 0.551** from position alone, and **0.721** when body orientation
is available (the published position+orientation reference on this dataset is
0.83).

THE DETECTOR — shared o-space with a transitivity guard
-------------------------------------------------------
An F-formation is not "people who are near each other". Kendon's definition
(1990) is a set of people whose *transactional segments* overlap in a shared
**o-space** — the patch of floor the group encloses and attends to. Two people
0.7 m apart back to back are not a group; two people 1.4 m apart facing each
other are. Position alone cannot tell those apart, which is the whole of the
0.385-vs-0.83 gap the validation measured.

So each candidate group here faces four tests, and a cluster is only emitted if
it passes all of them:

1. **Connectivity** — the group is connected in the graph of pairs within
   `eps_m`. (This is the one test DBSCAN also applies.)
2. **Transitivity guard** — *every* pair inside the group, not just adjacent
   ones, lies within `max_pair_m(k)`, a cap that grows with group size because
   a ring of k people around one o-space is physically wider for larger k. This
   is what stops one chain welding a whole room together, and it is what
   DBSCAN's single linkage structurally cannot do.
3. **Arrangement** — for k >= 3, the members must sit around a centre rather
   than along a line: the ratio of the nearest member's distance from the
   group's centre to the mean member distance must be at least
   `circularity_min`. A true F-formation leaves its o-space empty; a chain
   across a room has members standing in the middle of its own bounding circle.
4. **Shared o-space** — wherever facing is known (`facing.py`), one common
   o-space centre is fitted to the members' facing rays by least squares, and
   every facing member's ray must point within `facing_tol_rad` of it, at a
   plausible distance. This accepts every arrangement Kendon describes —
   vis-a-vis, L-shape, side-by-side, ring — because all of them converge on a
   shared space, and rejects back-to-back and facing-apart pairs, which do not.

Where facing is unknown — a stationary person, until the SAM-keypoint source in
`facing.py` is wired up — tests 1-3 still run and test 4 abstains. That is the
deliberate fallback: distance plus a transitivity guard, never distance alone,
and never "unknown" quietly treated as "forward".

Candidate groups are built **divisively**: each connected component of the
`eps_m` graph is tested whole, and a set that fails is cut along the weakest
seam of its affinity graph (the sign-ordered Fiedler vector of the normalised
Laplacian — a normalised minimum cut) and each half reconsidered. People who
end up alone are noise, exactly as DBSCAN's -1 label was. Divisive rather than
agglomerative because a partial arc of a genuine eight-person ring looks like a
chain on the way up, so merging stalls before it reaches the real group;
`build="agglomerative"` keeps that path selectable and scores within 0.001 F1
on SALSA. Deterministic, and O(component^3) at worst with components bounded by
the eps graph.

WHY THIS ALGORITHM, and what was rejected
-----------------------------------------
Considered and rejected: **full GCFF** (Setti et al. 2015 — a grid of candidate
o-space centres, each person casting a vote, labelled by graph cuts). It is the
strongest published method on SALSA, and it needs an orientation for *every*
person to vote with. In the scene Proxemix actually faces, most people are
standing still and have no motion facing at all, so every vote degenerates to an
isotropic ring and the graph cut reduces to distance clustering — with a
per-frame optimisation and a grid resolution parameter bolted on. Also
considered: **dominant sets** (Hung & Kroese, ICMI 2011), which does enforce
internal cohesion and would fix the chaining; rejected because its replicator
dynamics give no place to express the geometric tests above as hard constraints,
and its behaviour is tuned through one opaque affinity scale rather than through
parameters that are distances in metres a reader can argue about.

The merge-test formulation keeps the o-space idea, degrades gracefully to a
guarded distance method when facing is missing, and every threshold is a metre
or an angle.

`method="dbscan"` keeps the old single-linkage path available for comparison —
`outputs/salsa_validation.md` scores both.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# Defaults. Every one of these is a distance in metres, an angle, or a plain
# ratio, so it can be defended without reference to how it was fitted.
# ---------------------------------------------------------------------------

# Pair-linking radius: how far apart two people may be and still be *linked*.
# It is no longer the chain lever it was under DBSCAN — the transitivity guard
# below does that job now — so it is set from what it actually governs. Measured
# off SALSA's own ground-truth groups: the widest nearest-neighbour link inside
# a real conversation group is 0.90-1.26 m at the 90th percentile across both
# parts, so a 0.8 m radius cannot even connect a quarter of the true dyads.
# 1.1 m sits at the top of Hall's (1966) *personal distance* band (0.45-1.2 m),
# which `hall_bands.py` already encodes, and the F1 surface is flat from 1.0 to
# 1.2 m. The old default of 2.0 m was set against Grand Central Station, a
# sparse transit hall, and chains a dense indoor scene into one blob.
DEFAULT_EPS_M = 1.1

# The transitivity guard. cap(k) = MAX_PAIR_M + MAX_PAIR_GROW_M * (k - 2), so a
# dyad may span 1.5 m and an eight-person audience 1.5 + 6*0.15 = 2.4 m. The
# growth term is what lets a genuine large group be wide while still refusing a
# chain of the same width. Fitted against the member-to-member spans of SALSA's
# own ground-truth groups by size — measured medians 0.72/0.86 m at k=2 rising
# to 2.41 m at k=8 — and the F1 surface is flat across base 1.4-1.6 m with
# growth 0.12-0.15. See `outputs/salsa_validation.md`, the 2026-10-26 section.
DEFAULT_MAX_PAIR_M = 1.5
DEFAULT_MAX_PAIR_GROW_M = 0.15

# Arrangement test: min(member distance from centre) / mean(member distance).
# 1.0 is a perfect ring, 0.0 means somebody is standing on the group's own
# centre. Applies from k >= 3. Measured on SALSA's ground truth, real triples
# score 0.68-0.78 at the median and 0.48-0.51 at the 10th percentile, while a
# line of three scores 0.0 — which is why this test is worth 0.04 F1 on its own
# (0.489 without it, 0.551 with it).
DEFAULT_CIRCULARITY_MIN = 0.35

# Transactional-segment length: how far in front of a person their share of the
# o-space sits. ~0.45 m puts two people facing each other at the 0.9 m that is
# the middle of Hall's personal band. Only used to regularise the o-space fit
# when the facing rays are near-parallel (side-by-side arrangements), where the
# intersection is otherwise undetermined.
DEFAULT_OSPACE_RADIUS_M = 0.45
# How far off the fitted o-space centre a member's facing ray may point.
# 75 degrees: generous enough for annotation noise and for the fact that people
# in conversation turn away from the exact centre constantly, tight enough to
# reject back-to-back (180) and facing-apart (>120) arrangements. Swept against
# SALSA's annotated body orientation: F1 0.690 at 45 deg, 0.711 at 60, 0.721 at
# 75, 0.716 at 90 — a flat peak, so nothing hangs on the exact value.
DEFAULT_FACING_TOL_RAD = np.deg2rad(75.0)
# The fitted o-space must be a plausible distance from its members — rays that
# only meet five metres away are not enclosing a shared space.
DEFAULT_OSPACE_MAX_DIST_M = 2.0
# Facing below this confidence abstains rather than voting (facing.py sets 0.0
# for a stationary person).
DEFAULT_FACING_MIN_CONF = 0.3

DEFAULT_METHOD = "ospace"
# How candidate groups are built out of a connected component: "divisive" cuts a
# failing set along its weakest seam and reconsiders each half; "agglomerative"
# merges upward from singletons. Both honour the same four tests; the divisive
# build is the default because a partial arc of a genuine large ring looks like
# a chain on the way up, so agglomeration stalls on big groups.
DEFAULT_BUILD = "divisive"

# Persistence filter. The gap tolerance is in SECONDS and, when left as None, is
# derived from the data's own sampling stride — the defect that made
# debounce_frames=40 silently produce zero episodes on any dataset sampled more
# sparsely than Grand Central.
DEFAULT_MIN_DURATION_S = 10.0
DEFAULT_MIN_GAP_S = 2.0          # floor, for densely sampled data
DEFAULT_GAP_STRIDE_FACTOR = 2.5  # tolerate one dropped sample, with slack

# The pairwise facing veto is only computed for pairs within the transitivity
# cap at this many members — the largest group worth entertaining. Without it the
# horizon grows with the number of people in frame, which made the veto cost
# O(people^2) least-squares solves on a 330-person transit-hall frame.
_VETO_HORIZON_K = 12

# Weight of the regularising pull in the o-space least-squares fit (see
# `_fit_ospace`). Small relative to a member's ray constraint, so it only
# decides the direction the data leaves undetermined.
_OSPACE_PRIOR_W = 0.4

CLUSTER_COLUMNS = ["frame", "t", "group_key", "n_members"]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _pair_cap(k, max_pair_m, grow_m):
    return max_pair_m + grow_m * max(0, k - 2)


def _circularity(xy, centre):
    r = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
    m = r.mean()
    if m <= 1e-9:
        return 1.0
    return float(r.min() / m)


def _fit_ospace(xy, u, prior, prior_w=_OSPACE_PRIOR_W):
    """Least-squares shared o-space centre for members whose facing is known.

    Each facing member contributes the ray `p_i + s * u_i` (s >= 0); the centre
    minimises the summed squared perpendicular distance to those rays, which is
    the closed-form 2x2 solve below. `prior` (the members' centroid pushed one
    transactional segment along their mean facing) is added as one weak
    pseudo-observation so near-parallel rays — a side-by-side pair, where the
    intersection is genuinely undetermined — still produce a sensible centre
    rather than a numerical accident.
    """
    M = prior_w * np.eye(2)
    b = prior_w * np.asarray(prior, dtype=float)
    for p_i, u_i in zip(xy, u):
        P = np.eye(2) - np.outer(u_i, u_i)   # projection onto the ray's normal
        M += P
        b += P @ p_i
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:          # pragma: no cover - prior_w > 0 prevents it
        return np.asarray(prior, dtype=float)


def _max_facing_deviation(xy, u, centre):
    """Largest angle between a member's facing direction and the direction from
    that member to the fitted o-space centre."""
    v = centre[None, :] - xy
    phi = np.arctan2(v[:, 1], v[:, 0])
    theta = np.arctan2(u[:, 1], u[:, 0])
    d = np.abs(np.arctan2(np.sin(phi - theta), np.cos(phi - theta)))
    return float(d.max()), np.hypot(v[:, 0], v[:, 1])


class _Frame:
    """One frame's geometry, precomputed once so the merge loop is cheap."""

    def __init__(self, xy, facing, conf, p):
        self.xy = xy
        self.n = len(xy)
        self.p = p
        dx = xy[:, None, 0] - xy[None, :, 0]
        dy = xy[:, None, 1] - xy[None, :, 1]
        self.D = np.hypot(dx, dy)

        self.known = np.zeros(self.n, dtype=bool)
        self.u = np.zeros((self.n, 2))
        if facing is not None:
            self.known = np.isfinite(facing) & (conf >= p["facing_min_conf"])
            k = self.known
            if k.any():
                self.u[k, 0] = np.cos(facing[k])
                self.u[k, 1] = np.sin(facing[k])

        with np.errstate(over="ignore"):
            self.A = np.exp(-0.5 * (self.D / p["eps_m"]) ** 2)

        # Pairwise facing veto: two confidently-facing people who cannot share
        # an o-space are never put in one group, whatever their distance. Only
        # computed for pairs close enough to ever land in one group — the cap at
        # `_VETO_HORIZON_K` members — and solved for every such pair at once,
        # because a frame of a busy transit hall holds hundreds of people and a
        # Python loop over their pairs dominated the whole pipeline's runtime.
        self.veto = np.zeros((self.n, self.n), dtype=bool)
        if self.known.sum() >= 2:
            horizon = _pair_cap(_VETO_HORIZON_K, p["max_pair_m"],
                                p["max_pair_grow_m"])
            cand = np.triu(self.known[:, None] & self.known[None, :], 1)
            cand &= self.D <= horizon
            ii, jj = np.nonzero(cand)
            if len(ii):
                ok, dev = self._pair_ospace(ii, jj)
                bad_i, bad_j = ii[~ok], jj[~ok]
                self.veto[bad_i, bad_j] = True
                self.veto[bad_j, bad_i] = True
                w = np.exp(-0.5 * (dev[ok] / p["facing_tol_rad"]) ** 2)
                gi, gj = ii[ok], jj[ok]
                self.A[gi, gj] = self.A[gi, gj] * w
                self.A[gj, gi] = self.A[gi, gj]

        self.edge = (self.D <= p["eps_m"]) & ~self.veto
        np.fill_diagonal(self.edge, False)

    def _pair_ospace(self, ii, jj):
        """Test 4 for many pairs at once. Returns (ok, max_deviation) arrays.

        The same least-squares fit as `_fit_ospace`, written out for the 2x2
        case so it runs over whole arrays of pairs: M = w*I + P_i + P_j with
        P = I - u u^T, inverted in closed form.
        """
        p = self.p
        pi, pj = self.xy[ii], self.xy[jj]
        ui, uj = self.u[ii], self.u[jj]
        w = _OSPACE_PRIOR_W
        prior = 0.5 * (pi + pj) + p["ospace_radius_m"] * 0.5 * (ui + uj)

        def proj_apply(u, v):
            # (I - u u^T) v, per row
            return v - (u * v).sum(axis=1, keepdims=True) * u

        # M = w*I + (I - ui ui^T) + (I - uj uj^T)
        a = w + (1 - ui[:, 0] ** 2) + (1 - uj[:, 0] ** 2)
        d = w + (1 - ui[:, 1] ** 2) + (1 - uj[:, 1] ** 2)
        b = -(ui[:, 0] * ui[:, 1]) - (uj[:, 0] * uj[:, 1])
        rhs = w * prior + proj_apply(ui, pi) + proj_apply(uj, pj)
        det = a * d - b * b
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        cx = (d * rhs[:, 0] - b * rhs[:, 1]) / det
        cy = (a * rhs[:, 1] - b * rhs[:, 0]) / det
        c = np.stack([cx, cy], axis=1)

        dev = np.zeros(len(ii))
        dist = np.zeros(len(ii))
        for pos, u in ((pi, ui), (pj, uj)):
            v = c - pos
            phi = np.arctan2(v[:, 1], v[:, 0])
            theta = np.arctan2(u[:, 1], u[:, 0])
            dev = np.maximum(dev, np.abs(np.arctan2(np.sin(phi - theta),
                                                    np.cos(phi - theta))))
            dist = np.maximum(dist, np.hypot(v[:, 0], v[:, 1]))
        ok = (dev <= p["facing_tol_rad"]) & (dist <= p["ospace_max_dist_m"])
        return ok, dev

    def _ospace_ok(self, idx):
        """Test 4 for the facing members of `idx`. Returns (ok, max_deviation)."""
        p = self.p
        idx = np.asarray(idx)
        kn = idx[self.known[idx]]
        if len(kn) == 0:
            return True, 0.0
        xy = self.xy[kn]
        u = self.u[kn]
        prior = self.xy[idx].mean(axis=0) + p["ospace_radius_m"] * u.mean(axis=0)
        centre = _fit_ospace(xy, u, prior)
        dev, dists = _max_facing_deviation(xy, u, centre)
        if dev > p["facing_tol_rad"]:
            return False, dev
        if dists.max() > p["ospace_max_dist_m"]:
            return False, dev
        return True, dev

    # -- the four tests, applied to a candidate member list -----------------
    def accept(self, members):
        p = self.p
        idx = np.asarray(members)
        k = len(idx)
        sub_d = self.D[np.ix_(idx, idx)]

        # 2. transitivity guard — every pair, not just adjacent ones
        if sub_d.max() > _pair_cap(k, p["max_pair_m"], p["max_pair_grow_m"]):
            return False, 0.0
        # no mutually vetoed pair inside the group
        if self.veto[np.ix_(idx, idx)].any():
            return False, 0.0
        # 1. connectivity inside the eps graph
        if k > 1 and not self._connected(idx):
            return False, 0.0
        # 3. arrangement — members around a centre, not strung along a line
        if k >= 3 and p["circularity_min"] is not None:
            centre = self.xy[idx].mean(axis=0)
            if _circularity(self.xy[idx], centre) < p["circularity_min"]:
                return False, 0.0
        # 4. one shared o-space for every member whose facing is known
        if self.known[idx].any():
            ok, _ = self._ospace_ok(idx)
            if not ok:
                return False, 0.0

        if k == 1:
            return True, 0.0
        iu = np.triu_indices(k, 1)
        score = float(self.A[np.ix_(idx, idx)][iu].mean())
        if score < p["min_affinity"]:
            return False, 0.0
        return True, score

    def _connected(self, idx):
        sub = self.edge[np.ix_(idx, idx)]
        seen = {0}
        stack = [0]
        while stack:
            v_row = sub[stack.pop()]
            for v in np.flatnonzero(v_row):
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(idx)


def _components(edge):
    n = len(edge)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(np.triu(edge, 1))):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


_SPECTRAL_MAX_N = 60   # above this, bisect on positions instead (speed guard)


def _spectral_bisect(idx, A):
    """Split a member list in two along the weakest seam of its affinity graph —
    the sign of the Fiedler vector of the normalised Laplacian, which is the
    standard approximation to a normalised minimum cut.

    This is the splitter that makes the divisive build behave correctly on the
    case that matters: a genuine eight-person ring with one bystander attached
    is cut by peeling the bystander off, not by sawing the ring in half, because
    the bystander's single weak link is the cheapest cut.
    """
    n = len(idx)
    if n <= 2:
        return [idx[0]], idx[1:]
    sub = A[np.ix_(idx, idx)].astype(float).copy()
    np.fill_diagonal(sub, 0.0)
    if n > _SPECTRAL_MAX_N:
        return _position_bisect(idx, sub)
    deg = sub.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    L = np.eye(n) - sub * dinv[:, None] * dinv[None, :]
    try:
        _, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:          # pragma: no cover
        return [idx[0]], idx[1:]
    f = vecs[:, 1] * dinv
    order = np.argsort(f, kind="stable")
    f_sorted = f[order]
    # cut at the largest gap in the sorted Fiedler values, keeping both sides
    # non-empty — more stable than a sign split when the graph is lopsided.
    gaps = np.diff(f_sorted)
    cut = int(np.argmax(gaps)) + 1 if len(gaps) else 1
    left = [idx[i] for i in order[:cut]]
    right = [idx[i] for i in order[cut:]]
    if not left or not right:
        return [idx[0]], idx[1:]
    return left, right


def _position_bisect(idx, sub):
    """Fallback splitter for very large components: two-means on the members'
    own positions, seeded at the two furthest-apart members. Only reached in
    crowds far denser than any F-formation scene (see `_SPECTRAL_MAX_N`)."""
    a = int(np.argmax(sub.sum(axis=1)))
    b = int(np.argmin(sub[a]))
    left = [idx[i] for i in range(len(idx)) if sub[i, a] >= sub[i, b]]
    right = [idx[i] for i in range(len(idx)) if sub[i, a] < sub[i, b]]
    if not left or not right:
        return [idx[0]], idx[1:]
    return left, right


def _partition(idx, fr, out, depth=0):
    """Divisive build: a set that passes all four tests is a group; one that
    fails is cut along its weakest seam and each half reconsidered. Members that
    end up alone are noise, exactly as DBSCAN's label -1 was."""
    if len(idx) == 0:
        return
    # never reason about a disconnected set as one candidate
    if len(idx) > 1:
        sub_edge = fr.edge[np.ix_(idx, idx)]
        comps = _components(sub_edge)
        if len(comps) > 1:
            for c in comps:
                _partition([idx[i] for i in c], fr, out, depth + 1)
            return
    if len(idx) == 1:
        return
    ok, _ = fr.accept(idx)
    if ok:
        out.append(idx)
        return
    left, right = _spectral_bisect(idx, fr.A)
    _partition(left, fr, out, depth + 1)
    _partition(right, fr, out, depth + 1)


def _agglomerate(comp, fr):
    """Agglomerative build, kept selectable: merge the cluster pair with the
    highest mean pairwise affinity that still passes all four tests, until none
    does. Scored against the divisive build in outputs/salsa_validation.md."""
    clusters = {i: [i] for i in comp}
    while True:
        best, best_score = None, -np.inf
        keys = list(clusters)
        for ai in range(len(keys)):
            for bi in range(ai + 1, len(keys)):
                a, b = keys[ai], keys[bi]
                ma, mb = clusters[a], clusters[b]
                if not fr.edge[np.ix_(ma, mb)].any():
                    continue
                ok, score = fr.accept(ma + mb)
                if ok and score > best_score:
                    best, best_score = (a, b), score
        if best is None:
            break
        a, b = best
        clusters[a] = clusters[a] + clusters.pop(b)
    return [m for m in clusters.values() if len(m) > 1]


def _group_frame(xy, facing, conf, p, frame_obj=None):
    """Returns a list of member-index lists for one frame."""
    fr = frame_obj if frame_obj is not None else _Frame(xy, facing, conf, p)
    out = []
    for comp in _components(fr.edge):
        if len(comp) == 1:
            continue
        if p.get("build", "divisive") == "agglomerative":
            out.extend(_agglomerate(comp, fr))
        else:
            _partition(list(comp), fr, out)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def detect_frame_clusters(df, eps_m=DEFAULT_EPS_M, min_samples=2,
                          x_col="x_m", y_col="y_m",
                          method=DEFAULT_METHOD,
                          facing_col="facing_rad", facing_conf_col="facing_conf",
                          max_pair_m=DEFAULT_MAX_PAIR_M,
                          max_pair_grow_m=DEFAULT_MAX_PAIR_GROW_M,
                          circularity_min=DEFAULT_CIRCULARITY_MIN,
                          ospace_radius_m=DEFAULT_OSPACE_RADIUS_M,
                          facing_tol_rad=DEFAULT_FACING_TOL_RAD,
                          ospace_max_dist_m=DEFAULT_OSPACE_MAX_DIST_M,
                          facing_min_conf=DEFAULT_FACING_MIN_CONF,
                          min_affinity=0.0,
                          build=DEFAULT_BUILD,
                          diagnostics=False):
    """Returns a long DataFrame: frame, t, group_key (comma-joined sorted
    track_ids), n_members — one row per detected group per frame.

    `method="ospace"` (the default) runs the four-test detector described in the
    module docstring; `method="dbscan"` runs the original single-linkage DBSCAN,
    kept so the two can be scored side by side.

    Facing is read from `facing_col`/`facing_conf_col` when those columns are
    present (add them with `proxemix.facing.add_facing`). They are optional:
    with no facing at all the detector runs its distance-plus-transitivity
    fallback and never silently treats "unknown" as "forward".

    `diagnostics=True` adds `diameter_m`, `n_facing_known` and `affinity`
    columns. The four canonical columns above are always present and always
    first, so `detect_group_episodes` and `episodes_to_event_rows` are
    unaffected.
    """
    if method not in ("ospace", "dbscan"):
        raise ValueError(f"unknown method {method!r} — use 'ospace' or 'dbscan'")

    p = {
        "eps_m": eps_m, "max_pair_m": max_pair_m,
        "max_pair_grow_m": max_pair_grow_m, "circularity_min": circularity_min,
        "ospace_radius_m": ospace_radius_m, "facing_tol_rad": facing_tol_rad,
        "ospace_max_dist_m": ospace_max_dist_m,
        "facing_min_conf": facing_min_conf, "min_affinity": min_affinity,
        "build": build,
    }
    has_facing = facing_col in df.columns and facing_conf_col in df.columns

    rows = []
    for frame, g in df.groupby("frame", sort=True):
        if len(g) < max(2, min_samples):
            continue
        xy = g[[x_col, y_col]].to_numpy(dtype=float)
        ids = g["track_id"].to_numpy()
        t_val = g["t"].iloc[0]
        fr = None

        if method == "dbscan":
            labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(xy)
            member_sets = [np.flatnonzero(labels == lbl)
                           for lbl in sorted(set(labels)) if lbl != -1]
        else:
            if has_facing:
                facing = g[facing_col].to_numpy(dtype=float)
                conf = np.nan_to_num(g[facing_conf_col].to_numpy(dtype=float))
            else:
                facing = conf = None
            fr = _Frame(xy, facing, conf, p)
            member_sets = _group_frame(xy, facing, conf, p, frame_obj=fr)

        for idx in member_sets:
            idx = np.asarray(idx, dtype=int)
            if len(idx) < min_samples:
                continue
            row = {
                "frame": frame, "t": t_val,
                "group_key": ",".join(sorted(ids[idx].tolist())),
                "n_members": int(len(idx)),
            }
            if diagnostics:
                sub = xy[idx]
                d = np.hypot(sub[:, None, 0] - sub[None, :, 0],
                             sub[:, None, 1] - sub[None, :, 1])
                row["diameter_m"] = float(d.max())
                if fr is not None:
                    iu = np.triu_indices(len(idx), 1)
                    row["n_facing_known"] = int(fr.known[idx].sum())
                    row["affinity"] = float(fr.A[np.ix_(idx, idx)][iu].mean())
                else:
                    row["n_facing_known"] = 0
                    row["affinity"] = float("nan")
            rows.append(row)

    cols = CLUSTER_COLUMNS + (["diameter_m", "n_facing_known", "affinity"]
                              if diagnostics else [])
    return pd.DataFrame(rows, columns=cols)


def auto_gap_s(clusters_df, min_gap_s=DEFAULT_MIN_GAP_S,
               stride_factor=DEFAULT_GAP_STRIDE_FACTOR):
    """The gap tolerance the data itself implies: `stride_factor` times the
    median interval between consecutive samples, floored at `min_gap_s`.

    This is the fix for the defect the SALSA validation found. A fixed
    frame-count tolerance is only meaningful at the sampling stride it was
    chosen for: 40 frames is 1.6 s on Grand Central's 25 fps per-frame tracks
    and *less than one sample* on SALSA's 3-second grid, so on SALSA every
    sample opened a new episode and nothing ever reached the 10 s minimum —
    zero group episodes, no error raised. Counting seconds against the observed
    stride cannot fail that way on any dataset.
    """
    if clusters_df is None or len(clusters_df) == 0:
        return float(min_gap_s)
    t = np.unique(np.asarray(clusters_df["t"], dtype=float))
    if len(t) < 2:
        return float(min_gap_s)
    d = np.diff(t)
    d = d[d > 0]
    if not len(d):
        return float(min_gap_s)
    return float(max(min_gap_s, stride_factor * float(np.median(d))))


def detect_group_episodes(clusters_df, max_gap_s=None,
                          min_duration_s=DEFAULT_MIN_DURATION_S,
                          debounce_frames=None,
                          min_gap_s=DEFAULT_MIN_GAP_S,
                          stride_factor=DEFAULT_GAP_STRIDE_FACTOR):
    """Same debounce -> RLE pattern as the other event detectors, applied per
    exact group_key (the member set must match exactly; a one-person change in
    membership starts a new episode — deliberately strict).

    **The gap tolerance is in seconds.** `max_gap_s=None` derives it from the
    data's own sampling stride via `auto_gap_s()`, so the filter is safe on any
    dataset; pass a number to fix it.

    `debounce_frames` is accepted for backward compatibility and still means
    frames when given — callers that passed it keep their exact old behaviour —
    but it is no longer the default, because a frame count cannot be safe across
    sampling strides. New code should not use it.
    """
    empty = pd.DataFrame(columns=[
        "group_key", "n_members", "frame_start", "frame_end",
        "t_start", "t_end", "duration_s",
    ])
    if clusters_df is None or clusters_df.empty:
        return empty

    df = clusters_df.sort_values(["group_key", "frame"]).copy()
    grp = df.groupby("group_key", sort=False)
    if debounce_frames is not None:
        gap = grp["frame"].diff()
        tol = float(debounce_frames)
    else:
        gap = grp["t"].diff()
        tol = float(max_gap_s) if max_gap_s is not None else auto_gap_s(
            clusters_df, min_gap_s=min_gap_s, stride_factor=stride_factor)

    new_episode = gap.isna() | (gap > tol)
    df["episode_id"] = new_episode.groupby(df["group_key"]).cumsum()

    episodes = df.groupby(["group_key", "episode_id"], as_index=False).agg(
        n_members=("n_members", "first"),
        frame_start=("frame", "min"),
        frame_end=("frame", "max"),
        t_start=("t", "min"),
        t_end=("t", "max"),
    )
    episodes["duration_s"] = episodes["t_end"] - episodes["t_start"]
    episodes = episodes[episodes["duration_s"] >= min_duration_s]
    if episodes.empty:
        return empty
    return (episodes.drop(columns=["episode_id"])
            .sort_values("duration_s", ascending=False)
            .reset_index(drop=True))


def episodes_to_event_rows(episodes):
    rows = []
    for _, r in episodes.iterrows():
        rows.append({
            "type": "group", "actor_ids": r.group_key,
            "frame_start": int(r.frame_start), "frame_end": int(r.frame_end),
            "zone": None, "value": float(r.n_members),
        })
    return pd.DataFrame(rows)


def detector_description(eps_m=DEFAULT_EPS_M, min_samples=2,
                         method=DEFAULT_METHOD,
                         max_pair_m=DEFAULT_MAX_PAIR_M,
                         max_pair_grow_m=DEFAULT_MAX_PAIR_GROW_M,
                         min_duration_s=DEFAULT_MIN_DURATION_S):
    """One sentence naming the detector and its live thresholds, for the
    researcher export's ethogram. Keyword-named rather than read out of
    `__defaults__` by position, so adding a parameter cannot silently make the
    ethogram describe the wrong thing."""
    if method == "dbscan":
        detection = (f"single-linkage DBSCAN per frame (eps={eps_m} m, "
                     f"min_samples={min_samples})")
    else:
        detection = (
            f"per frame, shared-o-space grouping: pairs linked within {eps_m} m, "
            f"every pair inside a group within {max_pair_m}+{max_pair_grow_m}*(k-2) m, "
            f"members arranged around a centre rather than in a line, and facing "
            f"rays required to converge on one o-space wherever facing is known"
        )
    return (f"{detection}; the exact member set must then persist "
            f">= {min_duration_s:g}s to count as a group episode.")


def self_test():
    """Hand-built frames that name what the rebuild is for. Run:
    `py -m proxemix.groups`.

    1. A chain: six people in a line 0.7 m apart. DBSCAN at eps 0.8 returns one
       six-person group; the o-space detector must refuse it.
    2. A dyad with a bystander: two people 0.6 m apart and a third 0.75 m from
       one of them but 1.3 m from the other. DBSCAN welds all three; the
       o-space detector must return the pair.
    3. A ring: eight people around a 1.2 m o-space, all facing inward. Both
       must find it — the guard must not cost large groups.
    4. The same two positions facing each other and back to back: a group, and
       not a group.
    5. The persistence filter on a 3 s and a 0.04 s sampling stride.
    """
    def frame(xy, facing=None, ids=None):
        n = len(xy)
        d = pd.DataFrame({
            "track_id": ids or [f"{i:02d}" for i in range(n)],
            "frame": [0] * n, "t": [0.0] * n,
            "x_m": [p[0] for p in xy], "y_m": [p[1] for p in xy],
        })
        if facing is not None:
            d["facing_rad"] = facing
            d["facing_conf"] = [0.0 if f is None or not np.isfinite(f) else 1.0
                                for f in facing]
        return d

    results = {}

    chain = frame([(0.7 * i, 0.0) for i in range(6)])
    results["chain_dbscan"] = detect_frame_clusters(
        chain, method="dbscan")["n_members"].tolist()
    results["chain_ospace"] = detect_frame_clusters(
        chain, method="ospace")["n_members"].tolist()
    assert results["chain_dbscan"] == [6], results
    assert max(results["chain_ospace"] or [0]) <= 2, results

    bystander = frame([(0.0, 0.0), (0.6, 0.0), (1.25, 0.30)])
    results["bystander_dbscan"] = detect_frame_clusters(
        bystander, method="dbscan")["group_key"].tolist()
    results["bystander_ospace"] = detect_frame_clusters(
        bystander, method="ospace")["group_key"].tolist()
    assert results["bystander_dbscan"] == ["00,01,02"], results
    assert results["bystander_ospace"] == ["00,01"], results

    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring_xy = [(1.2 * np.cos(a), 1.2 * np.sin(a)) for a in ang]
    inward = [(a + np.pi) % (2 * np.pi) for a in ang]
    results["ring_ospace"] = detect_frame_clusters(
        frame(ring_xy, facing=inward), eps_m=1.0, method="ospace")["n_members"].tolist()
    results["ring_no_facing"] = detect_frame_clusters(
        frame(ring_xy), eps_m=1.0, method="ospace")["n_members"].tolist()
    assert results["ring_ospace"] == [8], results
    assert results["ring_no_facing"] == [8], results

    b2b = frame([(0.0, 0.0), (0.8, 0.0)], facing=[np.pi, 0.0])
    talking = frame([(0.0, 0.0), (0.8, 0.0)], facing=[0.0, np.pi])
    side = frame([(0.0, 0.0), (0.8, 0.0)], facing=[np.pi / 2, np.pi / 2])
    results["back_to_back"] = detect_frame_clusters(b2b)["group_key"].tolist()
    results["face_to_face"] = detect_frame_clusters(talking)["group_key"].tolist()
    results["side_by_side"] = detect_frame_clusters(side)["group_key"].tolist()
    assert results["back_to_back"] == [], results
    assert results["face_to_face"] == ["00,01"], results
    assert results["side_by_side"] == ["00,01"], (
        "side-by-side is a Kendon arrangement and must survive", results)

    def strided(stride_s, fps, n=10):
        return pd.DataFrame({
            "frame": [int(round(i * stride_s * fps)) for i in range(n)],
            "t": [i * stride_s for i in range(n)],
            "group_key": ["01,02"] * n, "n_members": [2] * n,
        })
    salsa_like = strided(3.0, 15.0)
    gc_like = strided(0.04, 25.0, n=400)
    results["auto_gap_salsa_s"] = auto_gap_s(salsa_like)
    results["auto_gap_gc_s"] = auto_gap_s(gc_like)
    results["episodes_salsa_default"] = len(detect_group_episodes(salsa_like))
    results["episodes_gc_default"] = len(detect_group_episodes(gc_like))
    results["episodes_salsa_legacy_40_frames"] = len(
        detect_group_episodes(salsa_like, debounce_frames=40))
    assert results["episodes_salsa_default"] == 1, results
    assert results["episodes_gc_default"] == 1, results
    assert results["episodes_salsa_legacy_40_frames"] == 0, (
        "the legacy frame-count path should still show the defect it had", results)
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2, default=str))
    print("\ngroups self-test PASSED — the chain and the bystander are refused, "
          "the eight-person ring survives with and without facing, back-to-back "
          "is vetoed while face-to-face and side-by-side are kept, and the "
          "seconds-based persistence filter works on both a 3 s and a 0.04 s "
          "sampling stride (the frame-count path still fails on SALSA's, which "
          "is the defect it had).")
