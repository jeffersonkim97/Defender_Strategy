
# stp_rrtstar.py
# Refactored from notebook to a reusable module.
# Provides an STP-RRT* (Space-Time Parallel RRT*) planner with robust collision checks
# against static buildings and dynamic sensor FOVs (via dmap.gen_cam).
#
# Dependencies: numpy, shapely, time, json, os, math, random
#
# Public API:
#   - run_and_save(map_in, cam_dict, dmap, planner_cfg, out_json_path) -> dict
#   - stp_rrtstar(map_in, cam_dict, dmap, x0, xf, cfg) -> dict
#
# Output schema (compatible with zero_sum_game_casadi reader):
# {
#   "path": [[x,y,t], ...],
#   "vertex": {"start_tree": [[x,y,t], ...], "goal_tree": {}},
#   "edge":   {"start_tree": [[parent_idx, child_idx], ...], "goal_tree": {}},
#   "computation_time": float,
#   "distance_cost": float,
#   "time_cost": float
# }
#
# Notes:
# - map_in["st"]: static polygons {"n": int, "0": [[x,y],...], ...}
# - cam_dict["n"]: number of sensors; cam_dict["spec"]["panspeed"], cam_dict["spec"]["fov"] (half-angle, radians)
# - dmap must provide dmap.gen_cam(i, t) -> shapely Polygon/MultiPolygon or dict with "FOV_Poly".
# - cfg keys (with sensible defaults below).
#
# © 2025

from __future__ import annotations
import os, json, time, math, random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from shapely.validation import explain_validity

import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

# -----------------------------
# Data classes & configuration
# -----------------------------

@dataclass
class PlannerConfig:
    map_size: Tuple[float, float]             # (W, H)
    v_max: float = 1.0                        # max translational speed [m/s]
    steer_step: float = 1.0                   # nominal spatial step for extension [m]
    t_step_min: float = 0.1                   # min time step to avoid degeneracy [s]
    t_step_max: float = 3.0                   # max time step per extension [s]
    vehicle_radius: float = 0.5               # collision radius [m]
    goal_radius: float = 1.0                  # acceptable goal proximity [m]
    rewire_radius: float = 3.0                # RRT* rewiring radius [m]
    max_iter: int = 20000                     # hard iteration cap (safety)
    comp_time_limit: float = 5.0              # wall-clock cap [s]
    seed: Optional[int] = None                # RNG seed
    # Dynamic-collision sampling controls
    safety_angle_frac: int = 6                # pan step = fov_half/safety_angle_frac
    min_dyn_samples: int = 3
    max_dyn_samples: int = 1500

@dataclass
class Accel:
    static_union_prep: Any = None
    max_pan_rate: float = 0.0
    fov_half: float = math.pi/3

# -----------------------------
# Utility helpers
# -----------------------------

def _euclid(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))

def _extract_fov_poly(gen_cam_output, idx: int):
    out = gen_cam_output
    if isinstance(out, BaseGeometry):
        return out
    if isinstance(out, dict):
        if 'FOV_Poly' in out and isinstance(out['FOV_Poly'], BaseGeometry):
            return out['FOV_Poly']
        key = str(idx)
        if key in out and isinstance(out[key], dict) and 'FOV_Poly' in out[key]:
            poly = out[key]['FOV_Poly']
            if isinstance(poly, BaseGeometry):
                return poly
    return None

def setup_collision_accel(map_in: Dict[str, Any], cam_dict: Dict[str, Any]) -> Accel:
    st = map_in.get("st", {})
    n_static = int(st.get("n", 0))
    static_polys = [Polygon(st[str(i)]) for i in range(n_static)]
    static_union = unary_union(static_polys) if n_static > 0 else None
    static_union_prep = prep(static_union) if static_union is not None else None

    spec = cam_dict.get("spec", {})
    pans = np.array(spec.get("panspeed", []), dtype=float)
    max_pan_rate = float(np.max(np.abs(pans))) if pans.size else 0.0
    fov_half = float(spec.get("fov", [math.pi/3])[0])  # default 60 deg half

    return Accel(static_union_prep=static_union_prep,
                 max_pan_rate=max_pan_rate,
                 fov_half=fov_half)

def _sanitize_and_inflate_fov(geom, inflate):
    if geom is None:
        return None
    try:
        g = geom
        if not g.is_valid:
            g = g.buffer(0)
            if not g.is_valid:
                return None
        if inflate > 0.0:
            g = g.buffer(float(inflate))
        return g
    except Exception:
        return None

def _fov_at(i, t, cam_dict, dmap):
    # If you patched to use cam_dict["gen_cam"], call that here; otherwise:
    return dmap.gen_cam(i, t) if dmap is not None and hasattr(dmap, "gen_cam") else None

def _segment_dynamic_clear_strict(p1, p2, t1, t2,
                                  cfg, cam_dict, dmap, accel,
                                  depth=0, max_depth=22,
                                  ds_tol=None, dtheta_tol=None):
    """
    Recursively verify no dynamic collision on [t1,t2] while attacker moves
    linearly from p1->p2. Guarantees: if returns True, no intersection
    with any FOV over the whole interval, up to the given tolerances.
    """
    # tolerances
    if ds_tol is None:
        ds_tol = max(0.25*cfg.vehicle_radius, 1e-3)      # attacker translation per leaf
    if dtheta_tol is None:
        fov_half = accel.fov_half if accel else (math.pi/3)
        dtheta_tol = max(fov_half/float(max(cfg.safety_angle_frac,1)), 1e-3)

    # 1) check endpoints (cheap early outs)
    n_cam = int(cam_dict.get("n", 0))
    for (x,y,t) in ((p1[0],p1[1],t1),(p2[0],p2[1],t2)):
        veh_pt = Point(x,y)
        for i in range(n_cam):
            fov_raw = _fov_at(i, t, cam_dict, dmap)
            fov = _extract_fov_poly(fov_raw, i)
            fov = _sanitize_and_inflate_fov(fov, inflate=cfg.vehicle_radius)
            if fov is not None and fov.intersects(veh_pt):
                return False  # definite hit

    # 2) if leaf small enough in both attacker motion and FOV rotation, accept
    dxy = float(np.linalg.norm(np.asarray(p2)-np.asarray(p1)))
    dt  = abs(t2 - t1)
    if accel and accel.max_pan_rate > 0.0:
        dtheta = accel.max_pan_rate * dt
    else:
        dtheta = 0.0

    if (dxy <= ds_tol and dtheta <= dtheta_tol) or depth >= max_depth:
        # Final sample at midpoint to catch mid-interval hits
        tm = 0.5*(t1+t2)
        xm = 0.5*(p1[0]+p2[0]); ym = 0.5*(p1[1]+p2[1])
        veh_pt = Point(xm, ym)
        for i in range(n_cam):
            fov_raw = _fov_at(i, tm, cam_dict, dmap)
            fov = _extract_fov_poly(fov_raw, i)
            fov = _sanitize_and_inflate_fov(fov, inflate=cfg.vehicle_radius)
            if fov is not None and fov.intersects(veh_pt):
                return False
        return True

    # 3) subdivide in time (and linearly in space)
    tm = 0.5*(t1+t2)
    pm = ((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5)

    # left half
    if not _segment_dynamic_clear_strict(p1, pm, t1, tm, cfg, cam_dict, dmap, accel,
                                         depth+1, max_depth, ds_tol, dtheta_tol):
        return False
    # right half
    if not _segment_dynamic_clear_strict(pm, p2, tm, t2, cfg, cam_dict, dmap, accel,
                                         depth+1, max_depth, ds_tol, dtheta_tol):
        return False
    return True

def validate_point(q: np.ndarray, cfg: PlannerConfig,
                   map_in: Dict[str, Any], cam_dict: Dict[str, Any], dmap,
                   accel: Optional[Accel]=None) -> bool:
    x, y, t = float(q[0]), float(q[1]), float(q[2])
    W, H = float(cfg.map_size[0]), float(cfg.map_size[1])
    if not (0.0 <= x <= W and 0.0 <= y <= H):
        return False

    # static as before
    disc = Point(x, y).buffer(float(cfg.vehicle_radius))
    if accel and accel.static_union_prep is not None:
        if accel.static_union_prep.intersects(disc):
            return False
    else:
        st = map_in.get("st", {})
        for i in range(int(st.get("n", 0))):
            if Polygon(st[str(i)]).intersects(disc):
                return False

    # dynamic: sanitize+inflate FOV once and test against *point* (already buffered)
    n_cam = int(cam_dict.get("n", 0))
    for i in range(n_cam):
        fov_obj = dmap.gen_cam(i, t) if dmap is not None else None
        fov_poly = _extract_fov_poly(fov_obj, i)
        fov_poly = _sanitize_and_inflate_fov(fov_poly, inflate=0.0)  # point already buffered
        if fov_poly is not None and fov_poly.intersects(disc):
            return False

    return True

def collision_free_segment(q1, q2, cfg, map_in, cam_dict, dmap, accel=None):
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    p1, p2 = q1[:2], q2[:2]
    t1, t2 = float(q1[2]), float(q2[2])

    # 1) Exact static sweep
    swept = LineString([tuple(p1), tuple(p2)]).buffer(float(cfg.vehicle_radius), cap_style=1)
    if accel and accel.static_union_prep is not None:
        if accel.static_union_prep.intersects(swept):
            return False
    else:
        st = map_in.get("st", {})
        polys = [Polygon(st[str(i)]) for i in range(int(st.get("n", 0)))]
        if polys and unary_union(polys).intersects(swept):
            return False

    # 2) STRICT dynamic check over the whole interval
    return _segment_dynamic_clear_strict(tuple(p1), tuple(p2), t1, t2, cfg, cam_dict, dmap, accel)


# -----------------------------
# Core STP-RRT* implementation
# -----------------------------

def stp_rrtstar(map_in: Dict[str, Any],
                cam_dict: Dict[str, Any],
                dmap,
                x0: Tuple[float,float],
                xf: Tuple[float,float],
                cfg: PlannerConfig) -> Dict[str, Any]:
    """
    Run a single-tree STP-RRT* in (x,y,t).
    - Time increases monotonically; edge time is constrained by v_max.
    - Loitering allowed (t can advance with small spatial move).
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    accel = setup_collision_accel(map_in, cam_dict)
    W, H = cfg.map_size

    # Node storage
    V: List[np.ndarray] = []                    # nodes as np.array([x,y,t])
    parent: List[int] = []                      # parent indices (-1 for root)
    cost: List[float] = []                      # path cost (time by default)

    # Edge list for export
    E: List[Tuple[int,int]] = []

    # Initialize root (t=0)
    q_start = np.array([float(x0[0]), float(x0[1]), 0.0], dtype=float)
    if not validate_point(q_start, cfg, map_in, cam_dict, dmap, accel):
        raise ValueError("Start state is in collision.")
    V.append(q_start); parent.append(-1); cost.append(0.0)

    q_goal_xy = np.array([float(xf[0]), float(xf[1])], dtype=float)
    goal_idx: Optional[int] = None

    t0_wall = time.monotonic()
    it = 0

    def time_up() -> bool:
        return (time.monotonic() - t0_wall) >= float(cfg.comp_time_limit)

    # Precompute straight-line heuristic time to set t bounds loosely
    line_len = _euclid(q_start[:2], q_goal_xy)
    t_line = line_len / max(cfg.v_max, 1e-9)
    t_upper = 1.5 * t_line + 10.0  # slack

    while it < cfg.max_iter and not time_up():
        it += 1

        # ----- Sample in space-time -----
        if random.random() < 0.05:
            # Goal bias in XY; sample time near straight-line schedule
            x_rand, y_rand = float(q_goal_xy[0]), float(q_goal_xy[1])
            t_rand = random.uniform(0.0, t_upper)
        else:
            x_rand = random.uniform(0.0, W)
            y_rand = random.uniform(0.0, H)
            t_rand = random.uniform(0.0, t_upper)

        q_rand = np.array([x_rand, y_rand, t_rand], dtype=float)

        # ----- Nearest neighbor (euclidean in XYZ with scaled time) -----
        # Scale time dimension to prefer chronological growth
        def nn_metric(qi: np.ndarray) -> float:
            dx = qi[0] - q_rand[0]
            dy = qi[1] - q_rand[1]
            dt = (qi[2] - q_rand[2]) * (cfg.v_max)  # scale time ~ distance by v_max
            return dx*dx + dy*dy + dt*dt

        idx_near = int(np.argmin([nn_metric(qi) for qi in V]))
        q_near = V[idx_near]

        # ----- Steer (respect v_max) -----
        # Propose spatial step
        dvec = q_rand[:2] - q_near[:2]
        dnorm = np.linalg.norm(dvec)
        if dnorm < 1e-9:
            # advance time a little to avoid zero-duration edges
            t_prop = q_near[2] + cfg.t_step_min
            q_prop = np.array([q_near[0], q_near[1], t_prop], dtype=float)
        else:
            step = min(cfg.steer_step, dnorm)
            p_new = q_near[:2] + (dvec / dnorm) * step
            # compute minimum time needed by speed limit
            dt_min = step / max(cfg.v_max, 1e-9)
            dt_prop = max(cfg.t_step_min, min(cfg.t_step_max, dt_min * 1.05))  # small slack
            t_prop = q_near[2] + dt_prop
            q_prop = np.array([p_new[0], p_new[1], t_prop], dtype=float)

        # Enforce monotonic time
        if q_prop[2] <= q_near[2] + 1e-9:
            q_prop[2] = q_near[2] + cfg.t_step_min

        # Bounds & point validity
        if not validate_point(q_prop, cfg, map_in, cam_dict, dmap, accel):
            continue
        # Segment validity
        if not collision_free_segment(q_near, q_prop, cfg, map_in, cam_dict, dmap, accel):
            continue

        # ----- Choose parent (RRT* local optimization) -----
        # Neighborhood (in XY) for rewiring
        nbr_idx = []
        for j, qj in enumerate(V):
            if qj[2] <= q_prop[2] + 1e-6:  # only allow parents not in the future
                if _euclid(qj[:2], q_prop[:2]) <= cfg.rewire_radius:
                    nbr_idx.append(j)
        # Default parent is nearest
        best_parent = idx_near
        best_cost = cost[idx_near] + (q_prop[2] - q_near[2])  # time accumulation
        for j in nbr_idx:
            if time_up(): break
            qj = V[j]
            # respect causality
            if q_prop[2] <= qj[2] + 1e-9: 
                continue
            if not collision_free_segment(qj, q_prop, cfg, map_in, cam_dict, dmap, accel):
                continue
            c = cost[j] + (q_prop[2] - qj[2])
            if c < best_cost - 1e-9:
                best_cost = c; best_parent = j

        # Add node
        idx_new = len(V)
        V.append(q_prop); parent.append(best_parent); cost.append(best_cost)
        E.append((best_parent, idx_new))

        # ----- Rewire neighbors through q_prop if beneficial -----
        for j in nbr_idx:
            if time_up(): break
            if j == best_parent: continue
            qj = V[j]
            if qj[2] <= q_prop[2] + 1e-9:
                # only allow forward-time edges
                continue
            # attempt rewire j via idx_new (must reverse time -> generally disallowed)
            # RRT* in space-time with monotone time disallows making parent younger than child.
            # So we skip rewiring children whose time < q_prop.time.
            continue

        # ----- Check goal proximity in XY and a feasible final edge -----
        if _euclid(q_prop[:2], q_goal_xy) <= cfg.goal_radius:
            # connect a short final edge to exact goal XY at same time
            q_final = np.array([q_goal_xy[0], q_goal_xy[1], q_prop[2]], dtype=float)
            if validate_point(q_final, cfg, map_in, cam_dict, dmap, accel) and \
               collision_free_segment(q_prop, q_final, cfg, map_in, cam_dict, dmap, accel):
                # Accept goal
                idx_goal = len(V)
                V.append(q_final); parent.append(idx_new); cost.append(best_cost + (q_final[2]-q_prop[2]))
                E.append((idx_new, idx_goal))
                goal_idx = idx_goal
                break

    # -----------------------------
    # Reconstruct & package result
    # -----------------------------
    t_total = time.monotonic() - t0_wall

    if goal_idx is None:
        # try to pick best near-goal node
        dists = [ _euclid(q[:2], q_goal_xy) for q in V ]
        idx_near_goal = int(np.argmin(dists))
        goal_idx = idx_near_goal

    # backtrack
    path_idxs = []
    cur = goal_idx
    visited = set()
    while cur != -1 and cur not in visited:
        visited.add(cur)
        path_idxs.append(cur)
        cur = parent[cur]
    path_idxs.reverse()
    path = [ V[i].tolist() for i in path_idxs ]

    # distances/time
    total_dist = 0.0
    for i in range(1, len(path_idxs)):
        p0 = V[path_idxs[i-1]][:2]; p1 = V[path_idxs[i]][:2]
        total_dist += _euclid(p0, p1)
    time_cost = V[path_idxs[-1]][2] - V[path_idxs[0]][2] if len(path_idxs) > 1 else 0.0

    result = {
        "path": path,
        "vertex": {"start_tree": [vi.tolist() for vi in V], "goal_tree": {}},  # goal_tree kept for schema compatibility
        "edge":   {"start_tree": [list(e) for e in E], "goal_tree": {}},
        "computation_time": float(t_total),
        "distance_cost": float(total_dist),
        "time_cost": float(time_cost),
        "terminated_reason": ("goal_reached" if _euclid(np.array(path[-1][:2]), q_goal_xy) <= cfg.goal_radius else "time_or_iter_cap")
    }
    return result

# -----------------------------
# High-level convenience API
# -----------------------------

def run_and_save(map_in: Dict[str, Any],
                 cam_dict: Dict[str, Any],
                 dmap,
                 x0,
                 xf,
                 cfg,
                 out_json_path: str) -> Dict[str, Any]:
    """
    Build PlannerConfig from dict, run planner, save JSON to out_json_path.
    """
    # cfg = PlannerConfig(**planner_cfg)
    # x0 = tuple(map_in.get("x0", (0.0, 0.0)))
    # xf = tuple(map_in.get("xf", (cfg.map_size[0], cfg.map_size[1])))

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    result = stp_rrtstar(map_in, cam_dict, dmap, x0, xf, cfg)
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=4)
    return result
