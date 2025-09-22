# traj_pod_metrics.py
# Self-contained utilities for PoD and false-alarm metrics along an STP-RRT* trajectory.
from __future__ import annotations
import math
from typing import Iterable, Optional, Dict, Sequence

import numpy as np

# ---------------- Geometry helpers (self-contained) ----------------

def _orientation(p, q, r):
    return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

def _on_segment(p, q, r):
    return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
            min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

def _segments_intersect(a, b, c, d):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 == 0 and _on_segment(a, c, b): return True
    if o2 == 0 and _on_segment(a, d, b): return True
    if o3 == 0 and _on_segment(c, a, d): return True
    if o4 == 0 and _on_segment(c, b, d): return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def _polygon_edges(poly):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i+1) % n]

def _los_clear(ax: float, ay: float, bx: float, by: float, all_edges) -> bool:
    """Brute-force LOS test against polygon edges."""
    a = (ax, ay); b = (bx, by)
    for e1, e2 in all_edges:
        if _segments_intersect(a, b, e1, e2):
            # allow touching at sensor origin
            if (abs(e1[0]-a[0])<1e-9 and abs(e1[1]-a[1])<1e-9) or (abs(e2[0]-a[0])<1e-9 and abs(e2[1]-a[1])<1e-9):
                continue
            return False
    return True

def _wrap_angle(a: float) -> float:
    # map to (-pi, pi]
    return (a + math.pi) % (2.0*math.pi) - math.pi


# ---------------- Default PoD model (no hard range limit) ----------------

def default_pod_model(rr: float, r0: float) -> float:
    """
    Per-sensor probability of detection as a function of range (no cutoff):
        P(r) = exp(-(r / r0)^2)

    rr : range
    r0 : decay scale (>0). If <= 0, returns 0.
    """
    if r0 <= 0.0:
        return 0.0
    # clamp very large rr for numerical safety is optional; exp(-(rr/r0)^2) already safe
    return math.exp(- (rr / r0)**2)


# ---------------- Core computation ----------------

def compute_cumulative_pod_over_trajectory(
    map_in: Dict,
    cam_dict: Dict,
    path_array: Iterable[Iterable[float]],   # (N,3) [x,y,t] or (N,2) [x,y]
    *,
    los_blocks: bool = True,
    pod_fn = None,                           # callable: (r, r0)->scalar in [0,1]
    time_weighted: bool = False,             # if True, compute Σ P_k Δt_k as C_int
    # --- False alarm configuration (choose ONE style or leave both None) ---
    fa_per_sensor_per_frame: Optional[Sequence[float]] = None,   # per-frame FA probabilities p_i
    fa_rate_per_sensor_per_sec: Optional[Sequence[float]] = None,# Poisson rates λ_i (per second)
    # Optional override for frame cadence if path has no time column:
    dt_if_missing_time: Optional[float] = None
) -> Dict:
    """
    Computes PoD metrics over the attacker trajectory and false-alarm metrics.
    This version **removes any finite range gating**. Detection depends on:
        • angular FOV (half-angle) AND
        • line-of-sight (if los_blocks=True),
      with per-sensor PoD decaying exponentially with range via P(r)=exp(-(r/r0)^2).

    r0 selection (in order of precedence):
        1) cam_dict['spec']['range_scale_per_cam'][i]  (per-sensor)
        2) cam_dict['spec']['range_scale']             (scalar, global)
        3) 0.25 * map_diagonal                         (fallback)

    Returned keys:
        'P_per_vertex' : array (K,)
        'C_sum'        : float (Σ_k P_k)
        'C_int'        : float or None (time-weighted Σ_k P_k Δt_k)
        'P_ever'       : float = 1 - Π_k (1 - P_k)
        't'            : array of times
        'false_alarm'  : dict with FA metrics (see below)
    """
    if pod_fn is None:
        pod_fn = default_pod_model

    # --- Map & buildings ---
    st = map_in['st']
    size = np.array(st['size'], dtype=float)  # [xmin, xmax, ymin, ymax]
    x0, y0 = float(size[0]), float(size[2])
    x_span = float(size[1] - size[0])
    y_span = float(size[3] - size[2])
    map_diag = float(np.hypot(x_span, y_span))

    buildings = []
    for i in range(int(st['n'])):
        poly = np.array(st[str(i)], dtype=float)
        if x0 != 0.0 or y0 != 0.0:
            poly[:,0] -= x0; poly[:,1] -= y0
        buildings.append([tuple(p) for p in poly])

    # Edge list for LOS
    all_edges = []
    for poly in buildings:
        for e1, e2 in _polygon_edges(poly):
            all_edges.append((e1, e2))

    # --- Cameras / sensors ---
    n = int(cam_dict['n'])
    cam_x = np.array(cam_dict['x'], dtype=float)
    cam_y = np.array(cam_dict['y'], dtype=float)
    if x0 != 0.0 or y0 != 0.0:
        cam_x -= x0; cam_y -= y0

    spec = cam_dict['spec']
    theta0   = np.array(spec['init_angle'], dtype=float)
    bounds   = np.array(spec['bound'], dtype=float)   # (n,2) [+max, -max]
    fov_half = float(spec['fov'][0])                  # radians (half-angle)
    # spec['fov'][1] (old max range) is ignored by design in this version.
    dt_cam   = float(spec['cam_time'][1])
    w        = np.array(spec['panspeed'], dtype=float)

    # r0 selection
    if isinstance(spec.get('range_scale', None), (int, float)):
        r0_global = float(spec['range_scale'])
    else:
        r0_global = 0.25 * map_diag
    r0_per_cam = None
    if 'range_scale_per_cam' in spec:
        try:
            r0_per_cam = [float(v) for v in spec['range_scale_per_cam']]
            if len(r0_per_cam) != n:
                r0_per_cam = None
        except Exception:
            r0_per_cam = None

    # per-camera pan limits + r0
    cams = []
    for i in range(n):
        lo = theta0[i] + bounds[i,1]
        hi = theta0[i] + bounds[i,0]
        if lo > hi: lo, hi = hi, lo
        r0_i = r0_per_cam[i] if r0_per_cam is not None else r0_global
        cams.append({
            "x": cam_x[i], "y": cam_y[i],
            "theta0": theta0[i], "lo": lo, "hi": hi,
            "fov_half": fov_half, "w": w[i],
            "r0": float(r0_i)
        })

    # --- Path (N,3 or N,2) ---
    path_arr = np.array(path_array, dtype=float)
    if path_arr.ndim != 2 or path_arr.shape[1] not in (2,3):
        raise ValueError("path_array must be (N,3) [x,y,t] or (N,2) [x,y].")

    if path_arr.shape[1] == 2:
        # Synthesize time column.
        if dt_if_missing_time is None:
            dt_if_missing_time = dt_cam
        t_col = np.arange(path_arr.shape[0], dtype=float) * float(dt_if_missing_time)
        path_arr = np.c_[path_arr, t_col]

    if x0 != 0.0 or y0 != 0.0:
        path_arr[:,0] -= x0; path_arr[:,1] -= y0

    # Sort by time
    ord_idx = np.argsort(path_arr[:,2])
    path_arr = path_arr[ord_idx]
    px, py, pt = path_arr[:,0], path_arr[:,1], path_arr[:,2]
    K = len(pt)
    t0_abs, t_end_abs = float(pt[0]), float(pt[-1])
    T = max(0.0, t_end_abs - t0_abs)

    # camera boresight at absolute time t (triangular bounce within [lo,hi])
    def cam_angle(c, t_abs: float) -> float:
        if abs(c["w"]) < 1e-12 or c["hi"] == c["lo"]:
            return c["theta0"]
        span = c["hi"] - c["lo"]
        # Use absolute time to avoid dependence on first vertex time shifts
        raw  = c["theta0"] + c["w"]*(t_abs - t0_abs)
        per  = 2.0*span
        off  = (raw - c["lo"]) % per
        return (c["lo"] + off) if (off <= span) else (c["hi"] - (off - span))

    # --- Per-vertex network PoD (NO RANGE GATING) ---
    Pk = np.zeros(K, dtype=float)
    for k in range(K):
        xk, yk, tk = float(px[k]), float(py[k]), float(pt[k])
        prod_not = 1.0
        for c in cams:
            dx = xk - c["x"]; dy = yk - c["y"]
            r  = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            facing  = cam_angle(c, tk)
            # Angular FOV gate only (no hard range cutoff)
            if abs(_wrap_angle(bearing - facing)) > c["fov_half"]:
                continue
            # LOS occlusion (optional)
            if los_blocks and not _los_clear(c["x"], c["y"], xk, yk, all_edges):
                continue
            # Exponential PoD with scale r0
            pc = float(pod_fn(r, c["r0"]))
            if not np.isfinite(pc): pc = 0.0
            pc = min(1.0, max(0.0, pc))
            prod_not *= (1.0 - pc)
        Pk[k] = 1.0 - prod_not

    # --- Cumulative (sum over vertices) ---
    C_sum = float(Pk.sum())

    # --- Time-weighted integral (optional) ---
    C_int = None
    if time_weighted:
        if K >= 2:
            dt_seg = np.diff(pt)
            # carry forward last dt for the last vertex (common discrete approx)
            dt_full = np.concatenate([dt_seg, dt_seg[-1:]])
            C_int = float(np.dot(Pk, dt_full))
        else:
            C_int = 0.0

    # --- True overall probability over time (ever detected at least once) ---
    # P_ever = 1 - Π_k (1 - Pk), computed in log-domain for stability
    p = np.clip(Pk, 0.0, 1.0)
    s = float(np.sum(np.log1p(-p)))   # <= 0
    P_ever = float(1.0 - math.exp(s)) # exp(s) ∈ (0,1]

    # --- False alarm: two input styles ---
    fa_out: Dict[str, float | int] = {}
    if fa_per_sensor_per_frame is not None and fa_rate_per_sensor_per_sec is not None:
        raise ValueError("Specify only one of fa_per_sensor_per_frame OR fa_rate_per_sensor_per_sec, not both.")

    if fa_per_sensor_per_frame is not None:
        # Independent per-frame probabilities; network per-frame FA prob:
        fa = np.asarray(fa_per_sensor_per_frame, dtype=float)
        fa = np.clip(fa, 0.0, 1.0)
        P_FA_frame_net = 1.0 - float(np.prod(1.0 - fa))
        # Use camera cadence to estimate number of frames across the run:
        dt_frame = dt_cam if dt_cam > 0 else (dt_if_missing_time if dt_if_missing_time else 1.0)
        K_frames = int(math.ceil(T / dt_frame)) + 1 if T > 0 else K  # fallback to K when T==0
        P_FA_overall = 1.0 - (1.0 - P_FA_frame_net)**K_frames
        E_FA_count   = K_frames * P_FA_frame_net
        fa_out = {
            "mode": "per_frame",
            "P_FA_frame_net": P_FA_frame_net,
            "K_frames": K_frames,
            "P_FA_overall": float(P_FA_overall),
            "E_FA_count": float(E_FA_count)
        }

    elif fa_rate_per_sensor_per_sec is not None:
        # Independent Poisson processes; superposition ⇒ λ_net = Σ λ_i
        lam = np.asarray(fa_rate_per_sensor_per_sec, dtype=float)
        lam = np.clip(lam, 0.0, np.inf)
        lambda_net = float(np.sum(lam))
        P_FA_overall = 1.0 - math.exp(-lambda_net * T)
        E_FA_count   = lambda_net * T
        fa_out = {
            "mode": "rate",
            "lambda_FA_net": lambda_net,
            "T": T,
            "P_FA_overall": float(P_FA_overall),
            "E_FA_count": float(E_FA_count)
        }

    # --- Package results ---
    return {
        "P_per_vertex": Pk,   # shape (K,)
        "C_sum": C_sum,       # Σ_k P_k
        "C_int": C_int,       # Σ_k P_k Δt_k if requested (else None)
        "P_ever": P_ever,     # 1 - Π_k (1 - P_k)
        "t": pt,              # times at vertices
        "false_alarm": fa_out # {} if no FA inputs provided
    }

# # traj_pod_metrics.py
# # Self-contained utilities for PoD and false-alarm metrics along an STP-RRT* trajectory.

# from __future__ import annotations
# import math
# from typing import Iterable, Optional, Dict, Sequence

# import numpy as np


# # ---------------- Geometry helpers (self-contained) ----------------

# def _orientation(p, q, r):
#     return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

# def _on_segment(p, q, r):
#     return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
#             min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

# def _segments_intersect(a, b, c, d):
#     o1 = _orientation(a, b, c)
#     o2 = _orientation(a, b, d)
#     o3 = _orientation(c, d, a)
#     o4 = _orientation(c, d, b)
#     if o1 == 0 and _on_segment(a, c, b): return True
#     if o2 == 0 and _on_segment(a, d, b): return True
#     if o3 == 0 and _on_segment(c, a, d): return True
#     if o4 == 0 and _on_segment(c, b, d): return True
#     return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# def _polygon_edges(poly):
#     n = len(poly)
#     for i in range(n):
#         yield poly[i], poly[(i+1) % n]

# def _los_clear(ax: float, ay: float, bx: float, by: float, all_edges) -> bool:
#     """
#     Brute-force LOS test against polygon edges.
#     """
#     a = (ax, ay); b = (bx, by)
#     for e1, e2 in all_edges:
#         if _segments_intersect(a, b, e1, e2):
#             # allow touching at sensor origin
#             if (abs(e1[0]-a[0])<1e-9 and abs(e1[1]-a[1])<1e-9) or (abs(e2[0]-a[0])<1e-9 and abs(e2[1]-a[1])<1e-9):
#                 continue
#             return False
#     return True

# def _wrap_angle(a: float) -> float:
#     # map to (-pi, pi]
#     return (a + math.pi) % (2.0*math.pi) - math.pi


# # ---------------- Default PoD model ----------------

# def default_pod_model(rr: float, Rmax: float) -> float:
#     """
#     Default per-sensor probability of detection as a function of range:
#         P = exp(-(r / (0.5*Rmax))^2)

#     Returns a scalar in [0, 1].
#     """
#     # guard for Rmax <= 0
#     if Rmax <= 0.0:
#         return 0.0
#     return math.exp(- (rr / (0.5*Rmax))**2)


# # ---------------- Core computation ----------------

# def compute_cumulative_pod_over_trajectory(
#     map_in: Dict,
#     cam_dict: Dict,
#     path_array: Iterable[Iterable[float]],   # (N,3) [x,y,t] or (N,2) [x,y]
#     *,
#     los_blocks: bool = True,
#     pod_fn = None,                           # callable: (r, Rmax)->scalar in [0,1]
#     time_weighted: bool = False,             # if True, compute Σ P_k Δt_k as C_int
#     # --- False alarm configuration (choose ONE style or leave both None) ---
#     fa_per_sensor_per_frame: Optional[Sequence[float]] = None,  # per-frame FA probabilities p_i
#     fa_rate_per_sensor_per_sec: Optional[Sequence[float]] = None, # Poisson rates λ_i (per second)
#     # Optional override for frame cadence if path has no time column:
#     dt_if_missing_time: Optional[float] = None
# ) -> Dict:
#     """
#     Computes PoD metrics over the attacker trajectory 'path_array' and false-alarm metrics.

#     Definitions
#     ----------
#     At vertex k with attacker at (x_k, y_k, t_k):
#       - For sensor i, after range/FOV/LOS gating with max range R_i and half-angle α_i:
#             P_{i,k} = p_i(r_{i,k}) ∈ [0,1], else 0.
#       - Network PoD at vertex k (independent sensors):
#             P_k = 1 - Π_i (1 - P_{i,k}).

#     Returned Metrics
#     ----------------
#     - 'P_per_vertex': np.ndarray shape (K,)
#           P_k for each path vertex (network PoD).
#     - 'C_sum': float
#           Σ_k P_k (cumulative PoD over vertices; not a probability).
#     - 'C_int': Optional[float]
#           If time_weighted=True (and times available), Σ_k P_k Δt_k (≈ time integral).
#     - 'P_ever': float
#           True probability of detection at least once over the run:
#               P_ever = 1 - Π_k (1 - P_k).
#     - 'false_alarm': dict with keys depending on input mode:
#           * If fa_per_sensor_per_frame specified:
#                 'mode' : 'per_frame'
#                 'P_FA_frame_net' : float = 1 - Π_i(1 - p_i)
#                 'K_frames' : int  ~ ceil((t_end - t0)/dt_cam) + 1
#                 'P_FA_overall' : float = 1 - (1 - P_FA_frame_net)^K_frames
#                 'E_FA_count'   : float = K_frames * P_FA_frame_net   (expected count)
#           * If fa_rate_per_sensor_per_sec specified:
#                 'mode' : 'rate'
#                 'lambda_FA_net' : float = Σ_i λ_i    (Poisson superposition)
#                 'T' : float = t_end - t0
#                 'P_FA_overall' : float = 1 - exp(-lambda_FA_net * T)
#                 'E_FA_count'   : float = lambda_FA_net * T
#           * If neither provided: empty dict {}.

#     Notes
#     -----
#     - Units: All angles are assumed **radians** in cam_dict.
#     - LOS is computed brute-force against all building polygon edges.
#     - If path has no time column and dt_if_missing_time is None, dt is taken from cam_dict['spec']['cam_time'][1].
#     """
#     if pod_fn is None:
#         pod_fn = default_pod_model

#     # --- Map & buildings ---
#     st = map_in['st']
#     size = np.array(st['size'], dtype=float)  # [xmin, xmax, ymin, ymax]
#     x0, y0 = float(size[0]), float(size[2])

#     buildings = []
#     for i in range(int(st['n'])):
#         poly = np.array(st[str(i)], dtype=float)
#         if x0 != 0.0 or y0 != 0.0:
#             poly[:,0] -= x0; poly[:,1] -= y0
#         buildings.append([tuple(p) for p in poly])

#     # Edge list for LOS
#     all_edges = []
#     for poly in buildings:
#         for e1, e2 in _polygon_edges(poly):
#             all_edges.append((e1, e2))

#     # --- Cameras / sensors ---
#     n = int(cam_dict['n'])
#     cam_x = np.array(cam_dict['x'], dtype=float)
#     cam_y = np.array(cam_dict['y'], dtype=float)
#     if x0 != 0.0 or y0 != 0.0:
#         cam_x -= x0; cam_y -= y0

#     spec = cam_dict['spec']
#     theta0   = np.array(spec['init_angle'], dtype=float)
#     bounds   = np.array(spec['bound'], dtype=float)   # (n,2) [+max, -max]
#     fov_half = float(spec['fov'][0])                  # radians (half-angle)
#     Rmax     = float(spec['fov'][1])                  # max range (assumed same for all sensors here)
#     dt_cam   = float(spec['cam_time'][1])
#     w        = np.array(spec['panspeed'], dtype=float)

#     # per-camera pan limits
#     cams = []
#     for i in range(n):
#         lo = theta0[i] + bounds[i,1]
#         hi = theta0[i] + bounds[i,0]
#         if lo > hi: lo, hi = hi, lo
#         cams.append({
#             "x": cam_x[i], "y": cam_y[i],
#             "theta0": theta0[i], "lo": lo, "hi": hi,
#             "fov_half": fov_half, "Rmax": Rmax, "w": w[i]
#         })

#     # --- Path (N,3 or N,2) ---
#     path_arr = np.array(path_array, dtype=float)
#     if path_arr.ndim != 2 or path_arr.shape[1] not in (2,3):
#         raise ValueError("path_array must be (N,3) [x,y,t] or (N,2) [x,y].")

#     if path_arr.shape[1] == 2:
#         # Synthesize time column.
#         if dt_if_missing_time is None:
#             dt_if_missing_time = dt_cam
#         t_col = np.arange(path_arr.shape[0], dtype=float) * float(dt_if_missing_time)
#         path_arr = np.c_[path_arr, t_col]

#     if x0 != 0.0 or y0 != 0.0:
#         path_arr[:,0] -= x0; path_arr[:,1] -= y0

#     # Sort by time
#     ord_idx = np.argsort(path_arr[:,2])
#     path_arr = path_arr[ord_idx]
#     px, py, pt = path_arr[:,0], path_arr[:,1], path_arr[:,2]
#     K = len(pt)
#     t0, t_end = float(pt[0]), float(pt[-1])
#     T = max(0.0, t_end - t0)

#     # camera boresight at absolute time t (triangular bounce within [lo,hi])
#     def cam_angle(c, t_abs: float) -> float:
#         if abs(c["w"]) < 1e-12 or c["hi"] == c["lo"]:
#             return c["theta0"]
#         span = c["hi"] - c["lo"]
#         raw  = c["theta0"] + c["w"]*(t_abs - t0)
#         per  = 2.0*span
#         off  = (raw - c["lo"]) % per
#         return (c["lo"] + off) if (off <= span) else (c["hi"] - (off - span))

#     # --- Per-vertex network PoD ---
#     Pk = np.zeros(K, dtype=float)
#     for k in range(K):
#         xk, yk, tk = float(px[k]), float(py[k]), float(pt[k])
#         prod_not = 1.0
#         for c in cams:
#             dx = xk - c["x"]; dy = yk - c["y"]
#             r  = math.hypot(dx, dy)
#             if r > c["Rmax"]:
#                 continue
#             bearing = math.atan2(dy, dx)
#             facing  = cam_angle(c, tk)
#             if abs(_wrap_angle(bearing - facing)) > c["fov_half"]:
#                 continue
#             if los_blocks and not _los_clear(c["x"], c["y"], xk, yk, all_edges):
#                 continue
#             pc = float(pod_fn(r, c["Rmax"]))
#             if not np.isfinite(pc): pc = 0.0
#             pc = min(1.0, max(0.0, pc))
#             prod_not *= (1.0 - pc)
#         Pk[k] = 1.0 - prod_not

#     # --- Cumulative (sum over vertices) ---
#     C_sum = float(Pk.sum())

#     # --- Time-weighted integral (optional) ---
#     C_int = None
#     if time_weighted:
#         if K >= 2:
#             dt_seg = np.diff(pt)
#             # carry forward last dt for the last vertex (common discrete approx)
#             dt_full = np.concatenate([dt_seg, dt_seg[-1:]])
#             C_int = float(np.dot(Pk, dt_full))
#         else:
#             C_int = 0.0

#     # --- True overall probability over time (ever detected at least once) ---
#     # P_ever = 1 - Π_k (1 - Pk), numerically stable via log1p
#     p = np.clip(Pk, 0.0, 1.0)
#     s = float(np.sum(np.log1p(-p)))
#     P_ever = float(1.0 - math.exp(s))  # s <= 0; exp(s) ∈ (0,1]

#     # --- False alarm: two input styles ---
#     fa_out: Dict[str, float | int] = {}
#     if fa_per_sensor_per_frame is not None and fa_rate_per_sensor_per_sec is not None:
#         raise ValueError("Specify only one of fa_per_sensor_per_frame OR fa_rate_per_sensor_per_sec, not both.")

#     if fa_per_sensor_per_frame is not None:
#         # Independent per-frame probabilities; network per-frame FA prob:
#         fa = np.asarray(fa_per_sensor_per_frame, dtype=float)
#         fa = np.clip(fa, 0.0, 1.0)
#         P_FA_frame_net = 1.0 - float(np.prod(1.0 - fa))
#         # Use camera cadence to estimate number of frames across the run:
#         dt_frame = dt_cam if dt_cam > 0 else (dt_if_missing_time if dt_if_missing_time else 1.0)
#         K_frames = int(math.ceil(T / dt_frame)) + 1 if T > 0 else K  # fallback to K when T==0
#         P_FA_overall = 1.0 - (1.0 - P_FA_frame_net)**K_frames
#         E_FA_count   = K_frames * P_FA_frame_net
#         fa_out = {
#             "mode": "per_frame",
#             "P_FA_frame_net": P_FA_frame_net,
#             "K_frames": K_frames,
#             "P_FA_overall": float(P_FA_overall),
#             "E_FA_count": float(E_FA_count)
#         }

#     elif fa_rate_per_sensor_per_sec is not None:
#         # Independent Poisson processes; superposition ⇒ λ_net = Σ λ_i
#         lam = np.asarray(fa_rate_per_sensor_per_sec, dtype=float)
#         lam = np.clip(lam, 0.0, np.inf)
#         lambda_net = float(np.sum(lam))
#         P_FA_overall = 1.0 - math.exp(-lambda_net * T)
#         E_FA_count   = lambda_net * T
#         fa_out = {
#             "mode": "rate",
#             "lambda_FA_net": lambda_net,
#             "T": T,
#             "P_FA_overall": float(P_FA_overall),
#             "E_FA_count": float(E_FA_count)
#         }

#     # --- Package results ---
#     return {
#         "P_per_vertex": Pk,   # shape (K,)
#         "C_sum": C_sum,       # Σ_k P_k  (your "cumulative PoD over vertices")
#         "C_int": C_int,       # Σ_k P_k Δt_k if requested (else None)
#         "P_ever": P_ever,     # 1 - Π_k (1 - P_k)
#         "t": pt,              # times at vertices
#         "false_alarm": fa_out # {} if no FA inputs provided
#     }
