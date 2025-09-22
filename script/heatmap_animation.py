# heatmap_animation.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patches import Polygon as PolyPatch, Wedge, FancyArrow

# ----------------- Geometry / LOS -----------------

def orientation(p, q, r):
    return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

def on_segment(p, q, r):
    return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
            min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

def segments_intersect(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 == 0 and on_segment(a, c, b): return True
    if o2 == 0 and on_segment(a, d, b): return True
    if o3 == 0 and on_segment(c, a, d): return True
    if o4 == 0 and on_segment(c, b, d): return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def polygon_edges(poly):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i+1) % n]

class EdgeIndex:
    """Uniform grid buckets for building edges to accelerate LOS."""
    def __init__(self,
                 all_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                 map_size: Tuple[float, float],
                 buckets_per_axis: int = 32):
        self.edges = all_edges
        self.width, self.height = float(map_size[0]), float(map_size[1])
        self.B = max(4, int(buckets_per_axis))
        self.cell_w = self.width / self.B
        self.cell_h = self.height / self.B
        self.buckets = [[[] for _ in range(self.B)] for _ in range(self.B)]
        self._build()

    def _build(self):
        for eid, (p, q) in enumerate(self.edges):
            xmin = int(min(p[0], q[0]) // self.cell_w)
            xmax = int(max(p[0], q[0]) // self.cell_w)
            ymin = int(min(p[1], q[1]) // self.cell_h)
            ymax = int(max(p[1], q[1]) // self.cell_h)
            xmin = max(0, xmin); xmax = min(self.B-1, xmax)
            ymin = max(0, ymin); ymax = min(self.B-1, ymax)
            for gx in range(xmin, xmax+1):
                for gy in range(ymin, ymax+1):
                    self.buckets[gy][gx].append(eid)

    def _bbox_cells(self, a, b):
        xmin = int(min(a[0], b[0]) // self.cell_w)
        xmax = int(max(a[0], b[0]) // self.cell_w)
        ymin = int(min(a[1], b[1]) // self.cell_h)
        ymax = int(max(a[1], b[1]) // self.cell_h)
        xmin = max(0, xmin); xmax = min(self.B-1, xmax)
        ymin = max(0, ymin); ymax = min(self.B-1, ymax)
        return xmin, xmax, ymin, ymax

    def candidate_edges_for_segment(self, a, b):
        xmin, xmax, ymin, ymax = self._bbox_cells(a, b)
        seen = set()
        out = []
        for gx in range(xmin, xmax+1):
            for gy in range(ymin, ymax+1):
                for eid in self.buckets[gy][gx]:
                    if eid not in seen:
                        seen.add(eid)
                        out.append(eid)
        return out

def los_is_clear_indexed(a, b, edge_index: EdgeIndex, all_edges) -> bool:
    for eid in edge_index.candidate_edges_for_segment(a, b):
        e1, e2 = all_edges[eid]
        if segments_intersect(a, b, e1, e2):
            # allow touching at start point (sensor on wall)
            if (abs(e1[0]-a[0])<1e-9 and abs(e1[1]-a[1])<1e-9) or (abs(e2[0]-a[0])<1e-9 and abs(e2[1]-a[1])<1e-9):
                continue
            return False
    return True


# ----------------- Detection helpers -----------------

def in_fov(dx, dy, facing, half_angle):
    ang = np.arctan2(dy, dx)
    diff = (ang - facing + math.pi) % (2.0 * math.pi) - math.pi
    return np.abs(diff) <= half_angle

def pod_exponential(range_grid, r0):
    """P(r) = exp(-(r/r0)^2)."""
    return np.exp(- (range_grid / r0) ** 2)


# ----------------- Main API -----------------

def generate_heatmap(
    map_in: Dict,
    cam_dict: Dict,
    path_array: Iterable[Iterable[float]],   # (N,3) [x,y,t] or (N,2) [x,y]
    *,
    los_blocks: bool = True,
    los_buckets_per_axis: int = 32,
    grid_resolution: Tuple[int, int] = (100, 100),
    pod_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,

    # analytics controls
    snapshot_t: Optional[float] = None,      # seconds (clamped to run interval)
    compute_mean: bool = False,              # arithmetic time-average over run
    compute_union: bool = False,             # temporal union over run: 1 - Π_k(1-P_k)

    # normalization for displayed/saved grids: None | 'max' | 'minmax' | 'zscore'
    normalize: Optional[str] = None,

    # WHAT to display in the notebook: 'animation' | 'snapshot' | 'mean' | 'union'
    display: str = "animation",

    # optional RRT tree overlay
    draw_rrt: bool = False,
    Va: Optional[Iterable[Iterable[float]]] = None,
    Ea: Optional[Iterable[Iterable[int]]] = None,
    Vb: Optional[Iterable[Iterable[float]]] = None,
    Eb: Optional[Iterable[Iterable[int]]] = None,
    run_name: str = "stp-rrt*"
):
    """
    Renders Net PoD either as an animation or as a requested static heatmap (snapshot/mean/union).
    Saves PNGs to image/heatmap/. Only the selected `display` figure is shown.

    Changes from baseline:
      • No finite range limitation. Detection uses angular FOV + LOS only.
      • PoD decays exponentially with range via P(r)=exp(-(r/r0)^2).
      • FOV wedge drawn to map diagonal for visualization.

    Returns:
      - If display='animation': (fig_anim, anim, payload)
      - Else (snapshot/mean/union): (fig_static, payload)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    from matplotlib.patches import Polygon as PolyPatch, Wedge, FancyArrow

    # ---------- helpers ----------
    def _apply_norm(G: np.ndarray, mode: Optional[str]) -> np.ndarray:
        if mode is None:
            return G
        if mode == "max":
            m = float(np.nanmax(G));  return G / m if m > 0 else G
        if mode == "minmax":
            gmin = float(np.nanmin(G)); gmax = float(np.nanmax(G))
            return (G - gmin) / (gmax - gmin) if gmax > gmin else np.zeros_like(G)
        if mode == "zscore":
            mu = float(np.nanmean(G)); sd = float(np.nanstd(G))
            return (G - mu) / sd if sd > 0 else G - mu
        raise ValueError("normalize must be one of None, 'max', 'minmax', 'zscore'")

    def _save_png(fig, basename: str):
        out_dir = os.path.join("image", "heatmap")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, basename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    # ---------- map ----------
    st = map_in['st']
    size = np.array(st['size'], dtype=float)     # [xmin, xmax, ymin, ymax]
    x_map = float(size[1] - size[0])
    y_map = float(size[3] - size[2])
    map_diag = float(np.hypot(x_map, y_map))
    buildings = []
    for i in range(int(st['n'])):
        poly = np.array(st[str(i)], dtype=float)
        if size[0] != 0.0 or size[2] != 0.0:
            poly[:,0] -= size[0]; poly[:,1] -= size[2]
        buildings.append([tuple(p) for p in poly])

    # ---------- cameras ----------
    n = int(cam_dict['n'])
    cam_x = np.array(cam_dict['x'], dtype=float)
    cam_y = np.array(cam_dict['y'], dtype=float)
    spec = cam_dict['spec']
    init_angle = [float(a) for a in spec['init_angle']]
    bound_arr  = np.array(spec['bound'], dtype=float)        # (n,2): [+max, -max]
    fov_ang    = float(spec['fov'][0])                       # radians (half-angle)
    # fov range is ignored for detection; keep for compatibility if present
    num_frames = int(spec['cam_time'][0])
    dt         = float(spec['cam_time'][1])
    panspeed   = [float(w) for w in spec['panspeed']]

    # Optional range scale for exponential decay
    # If not provided, default r0 = 0.25 * map diagonal.
    if isinstance(spec.get('range_scale', None), (int, float)):
        default_r0 = float(spec['range_scale'])
    else:
        default_r0 = 0.25 * map_diag

    if size[0] != 0.0 or size[2] != 0.0:
        cam_x = cam_x - size[0]; cam_y = cam_y - size[2]

    cameras = []
    for i in range(n):
        r0_i = default_r0
        # allow per-sensor override if provided
        if 'range_scale_per_cam' in spec:
            try:
                r0_i = float(spec['range_scale_per_cam'][i])
            except Exception:
                r0_i = default_r0
        cameras.append({
            "x": cam_x[i], "y": cam_y[i],
            "theta0": init_angle[i],
            "bound_plus": bound_arr[i,0], "bound_minus": bound_arr[i,1],
            "fov_half": fov_ang, "panspeed": panspeed[i],
            "r0": r0_i
        })

    # ---------- path ----------
    path_arr = np.array(path_array, dtype=float)
    if path_arr.ndim != 2 or path_arr.shape[1] not in (2,3):
        raise ValueError("path_array must have shape (N,3) [x,y,t] or (N,2) [x,y].")
    if path_arr.shape[1] == 2:
        t_col = np.arange(path_arr.shape[0], dtype=float) * dt
        path_arr = np.c_[path_arr, t_col]
    if size[0] != 0.0 or size[2] != 0.0:
        path_arr[:,0] -= size[0]; path_arr[:,1] -= size[2]
    order = np.argsort(path_arr[:,2])
    path_arr = path_arr[order]
    px, py, pt = path_arr[:,0], path_arr[:,1], path_arr[:,2]
    t0, t_end = float(pt[0]), float(pt[-1])

    # ---------- timelines ----------
    frames_path = int(np.ceil((t_end - t0) / dt)) + 1
    frames_anim = max(num_frames, frames_path)
    frames_mean = frames_path

    t_frames = np.arange(frames_anim) * dt + t0
    target_x = np.interp(t_frames, pt, px)
    target_y = np.interp(t_frames, pt, py)

    # ---------- grid ----------
    nx, ny = int(grid_resolution[0]), int(grid_resolution[1])
    gx = np.linspace(0.0, x_map, nx); gy = np.linspace(0.0, y_map, ny)
    XX, YY = np.meshgrid(gx, gy)

    # ---------- edges & spatial index ----------
    all_edges = []
    for poly in buildings:
        for e1, e2 in polygon_edges(poly):
            all_edges.append((e1, e2))
    edge_index = EdgeIndex(all_edges, (x_map, y_map), buckets_per_axis=los_buckets_per_axis)

    # ---------- per-camera range grids ----------
    range_grids = [np.hypot(XX - c["x"], YY - c["y"]) for c in cameras]

    # ---------- pan model ----------
    def camera_angle_at_frame(cam, k):
        theta0 = cam["theta0"]; w = cam["panspeed"]
        lo = theta0 + cam["bound_minus"]; hi = theta0 + cam["bound_plus"]
        if lo > hi: lo, hi = hi, lo
        if abs(w) < 1e-12 or hi == lo: return theta0
        span = hi - lo; raw = theta0 + w*(k*dt); period = 2.0*span
        off = (raw - lo) % period
        return (lo + off) if (off <= span) else (hi - (off - span))

    # ---------- LOS masks (no range gating; LOS everywhere) ----------
    los_masks = []
    for cam, rr in zip(cameras, range_grids):
        if los_blocks:
            los = np.zeros_like(rr, dtype=bool)
            ii, jj = np.indices(rr.shape)
            ii = ii.ravel(); jj = jj.ravel()
            spt = (cam["x"], cam["y"])
            for a, b in zip(ii, jj):
                dpt = (float(XX[a, b]), float(YY[a, b]))
                if los_is_clear_indexed(spt, dpt, edge_index, all_edges):
                    los[a, b] = True
        else:
            los = np.ones_like(rr, dtype=bool)
        los_masks.append(los)

    # Default PoD function if none provided: exponential with r0
    if pod_fn is None:
        def pod_fn(rr, r0):  # noqa: F811
            return pod_exponential(rr, r0=r0)

    # ---------- per-frame network PoD ----------
    def network_pod_grid(k):
        prod_not = np.ones_like(XX, dtype=float)
        for cam, rr, los in zip(cameras, range_grids, los_masks):
            facing  = camera_angle_at_frame(cam, k)
            fov_ok  = in_fov(XX - cam["x"], YY - cam["y"], facing, cam["fov_half"])
            # No range_ok; detection is not hard-limited by range.
            mask    = fov_ok & los
            P_cam   = pod_fn(rr, cam["r0"])
            P_cam   = np.where(mask, P_cam, 0.0)
            prod_not *= (1.0 - P_cam)
        return 1.0 - prod_not

    # ---------- compute requested analytics ----------
    payload = {}

    if snapshot_t is not None or display == "snapshot":
        t_snap = float(np.clip(snapshot_t if snapshot_t is not None else t0, t0, t_end))
        k_snap = int(round((t_snap - t0) / dt))
        k_snap = int(np.clip(k_snap, 0, frames_anim - 1))
        P_snap = _apply_norm(network_pod_grid(k_snap), normalize)
        payload["snapshot"] = {"t": k_snap*dt + t0, "k": k_snap, "P": P_snap, "X": XX, "Y": YY}

    if compute_mean or display == "mean":
        mean_grid = np.zeros_like(XX, dtype=float)
        for k in range(frames_mean):
            mean_grid += network_pod_grid(k)
        mean_grid /= float(frames_mean)
        mean_grid = _apply_norm(mean_grid, normalize)
        payload["mean"] = {"P_mean": mean_grid, "X": XX, "Y": YY}

    if compute_union or display == "union":
        prod_not = np.ones_like(XX, dtype=float)
        for k in range(frames_mean):
            prod_not *= (1.0 - network_pod_grid(k))
        union_grid = 1.0 - prod_not
        union_grid = _apply_norm(union_grid, normalize)
        payload["union"] = {"P_union": union_grid, "X": XX, "Y": YY}

    # ---------- decorate helper (no grid, white dashed solver path) ----------
    def _decorate_axes(ax, title_suffix: str):
        # buildings
        for poly in buildings:
            ax.add_patch(PolyPatch(poly, closed=True, fill=True, alpha=0.25, edgecolor="black"))
        # sensors
        ax.scatter([c["x"] for c in cameras], [c["y"] for c in cameras], marker="^", s=60)
        # optional trees
        if draw_rrt:
            _maybe_draw_rrt(ax, Va, Ea, "tab:blue")
            _maybe_draw_rrt(ax, Vb, Eb, "tab:green")
        # full STP-RRT* path (white dashed)
        ax.plot(px, py, "--", color="white", lw=1.5, alpha=0.8)
        ax.set_xlim(0, x_map); ax.set_ylim(0, y_map)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # ---------- build exactly ONE figure to display ----------
    vmin = 0.0 if normalize in (None, "max", "minmax") else None
    vmax = 1.0 if normalize in (None, "max", "minmax") else None
    cbar_suffix = "" if normalize is None else f" [{normalize}]"

    if display == "animation":
        fig, ax = plt.subplots(figsize=(7, 7), dpi=250)
        im = ax.imshow(np.zeros_like(XX), origin="lower",
                       extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("P(detect)" + cbar_suffix)

        # initial wedges + heading (radius = map diagonal for visualization)
        fov_wedges = []
        for c in cameras:
            theta_deg = math.degrees(c["theta0"])
            wedge = Wedge(center=(c["x"], c["y"]),
                          r=map_diag,
                          theta1=theta_deg - math.degrees(c["fov_half"]),
                          theta2=theta_deg + math.degrees(c["fov_half"]),
                          alpha=0.08)
            ax.add_patch(wedge); fov_wedges.append(wedge)
            ax.add_patch(FancyArrow(c["x"], c["y"],
                                    2.0*math.cos(math.radians(theta_deg)),
                                    2.0*math.sin(math.radians(theta_deg)),
                                    width=0.25, length_includes_head=True, alpha=0.6))

        _decorate_axes(ax, "")

        target_dot, = ax.plot([], [], "ro", ms=5)

        def init():
            im.set_data(np.zeros_like(XX))
            target_dot.set_data([], [])
            return [im, target_dot] + fov_wedges

        def update(k):
            Pnet = _apply_norm(network_pod_grid(k), normalize)
            im.set_data(Pnet)
            for c, w in zip(cameras, fov_wedges):
                theta_deg = math.degrees(camera_angle_at_frame(c, k))
                w.set_theta1(theta_deg - math.degrees(c["fov_half"]))
                w.set_theta2(theta_deg + math.degrees(c["fov_half"]))
            kk = min(k, len(target_x)-1)
            target_dot.set_data(target_x[kk], target_y[kk])
            return [im, target_dot] + fov_wedges

        anim = FuncAnimation(fig, update, frames=frames_anim, init_func=init, blit=True, interval=50)
        _ = _save_png(fig, f"{run_name}_animation_frame0.png")
        return fig, anim, payload

    # Static outputs:
    fig_s, ax_s = plt.subplots(figsize=(7, 7), dpi=250)

    if display == "snapshot":
        P = payload["snapshot"]["P"]
        im = ax_s.imshow(P, origin="lower", extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax_s); cbar.set_label("Pd")
        _decorate_axes(ax_s, f"snapshot t={payload['snapshot']['t']:.2f}")
        _ = _save_png(fig_s, f"{run_name}_snapshot.png")
        return fig_s, payload

    if display == "mean":
        P = payload["mean"]["P_mean"]
        im = ax_s.imshow(P, origin="lower", extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax_s); cbar.set_label("Pd")
        _decorate_axes(ax_s, "time-average")
        _ = _save_png(fig_s, f"{run_name}_mean.png")
        return fig_s, payload

    if display == "union":
        P = payload["union"]["P_union"]
        im = ax_s.imshow(P, origin="lower", extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax_s); cbar.set_label("Pd")
        _decorate_axes(ax_s, "temporal-union")
        _ = _save_png(fig_s, f"{run_name}_union.png")
        return fig_s, payload

    raise ValueError("display must be one of 'animation', 'snapshot', 'mean', 'union'")

# ----------------- Helpers -----------------

def _maybe_draw_rrt(ax, V: Optional[Iterable[Iterable[float]]], E: Optional[Iterable[Iterable[int]]],
                    color: str, label: Optional[str] = None):
    if V is None or E is None:
        return
    V = np.array(V, dtype=float)
    if V.shape[1] >= 2:
        x, y = V[:,0], V[:,1]
    else:
        return
    ax.scatter(x, y, s=4, alpha=0.3, color=color, zorder=2)
    for e in E:
        try:
            i, j = int(e[0]), int(e[1])
            ax.plot([x[i], x[j]], [y[i], y[j]], lw=0.5, alpha=0.3, color=color, zorder=1)
        except Exception:
            continue


# # heatmap_animation.py
# from __future__ import annotations
# import math
# from typing import Dict, List, Tuple, Optional, Callable, Iterable

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, writers
# from matplotlib.patches import Polygon as PolyPatch, Wedge, FancyArrow

# # ----------------- Geometry / LOS -----------------

# def orientation(p, q, r):
#     return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

# def on_segment(p, q, r):
#     return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
#             min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

# def segments_intersect(a, b, c, d):
#     o1 = orientation(a, b, c)
#     o2 = orientation(a, b, d)
#     o3 = orientation(c, d, a)
#     o4 = orientation(c, d, b)
#     if o1 == 0 and on_segment(a, c, b): return True
#     if o2 == 0 and on_segment(a, d, b): return True
#     if o3 == 0 and on_segment(c, a, d): return True
#     if o4 == 0 and on_segment(c, b, d): return True
#     return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# def polygon_edges(poly):
#     n = len(poly)
#     for i in range(n):
#         yield poly[i], poly[(i+1) % n]

# class EdgeIndex:
#     """Uniform grid buckets for building edges to accelerate LOS."""
#     def __init__(self,
#                  all_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
#                  map_size: Tuple[float, float],
#                  buckets_per_axis: int = 32):
#         self.edges = all_edges
#         self.width, self.height = float(map_size[0]), float(map_size[1])
#         self.B = max(4, int(buckets_per_axis))
#         self.cell_w = self.width / self.B
#         self.cell_h = self.height / self.B
#         self.buckets = [[[] for _ in range(self.B)] for _ in range(self.B)]
#         self._build()

#     def _build(self):
#         for eid, (p, q) in enumerate(self.edges):
#             xmin = int(min(p[0], q[0]) // self.cell_w)
#             xmax = int(max(p[0], q[0]) // self.cell_w)
#             ymin = int(min(p[1], q[1]) // self.cell_h)
#             ymax = int(max(p[1], q[1]) // self.cell_h)
#             xmin = max(0, xmin); xmax = min(self.B-1, xmax)
#             ymin = max(0, ymin); ymax = min(self.B-1, ymax)
#             for gx in range(xmin, xmax+1):
#                 for gy in range(ymin, ymax+1):
#                     self.buckets[gy][gx].append(eid)

#     def _bbox_cells(self, a, b):
#         xmin = int(min(a[0], b[0]) // self.cell_w)
#         xmax = int(max(a[0], b[0]) // self.cell_w)
#         ymin = int(min(a[1], b[1]) // self.cell_h)
#         ymax = int(max(a[1], b[1]) // self.cell_h)
#         xmin = max(0, xmin); xmax = min(self.B-1, xmax)
#         ymin = max(0, ymin); ymax = min(self.B-1, ymax)
#         return xmin, xmax, ymin, ymax

#     def candidate_edges_for_segment(self, a, b):
#         xmin, xmax, ymin, ymax = self._bbox_cells(a, b)
#         seen = set()
#         out = []
#         for gx in range(xmin, xmax+1):
#             for gy in range(ymin, ymax+1):
#                 for eid in self.buckets[gy][gx]:
#                     if eid not in seen:
#                         seen.add(eid)
#                         out.append(eid)
#         return out

# def los_is_clear_indexed(a, b, edge_index: EdgeIndex, all_edges) -> bool:
#     for eid in edge_index.candidate_edges_for_segment(a, b):
#         e1, e2 = all_edges[eid]
#         if segments_intersect(a, b, e1, e2):
#             # allow touching at start point (sensor on wall)
#             if (abs(e1[0]-a[0])<1e-9 and abs(e1[1]-a[1])<1e-9) or (abs(e2[0]-a[0])<1e-9 and abs(e2[1]-a[1])<1e-9):
#                 continue
#             return False
#     return True


# # ----------------- Detection helpers -----------------

# def in_fov(dx, dy, facing, half_angle):
#     ang = np.arctan2(dy, dx)
#     diff = (ang - facing + math.pi) % (2.0 * math.pi) - math.pi
#     return np.abs(diff) <= half_angle

# def pod_exponential(range_grid, r0):
#     return np.exp(- (range_grid / r0) ** 2)


# # ----------------- Main API -----------------

# def generate_heatmap(
#     map_in: Dict,
#     cam_dict: Dict,
#     path_array: Iterable[Iterable[float]],   # (N,3) [x,y,t] or (N,2) [x,y]
#     *,
#     los_blocks: bool = True,
#     los_buckets_per_axis: int = 32,
#     grid_resolution: Tuple[int, int] = (100, 100),
#     pod_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,

#     # analytics controls
#     snapshot_t: Optional[float] = None,      # seconds (clamped to run interval)
#     compute_mean: bool = False,              # arithmetic time-average over run
#     compute_union: bool = False,             # temporal union over run: 1 - Π_k(1-P_k)

#     # normalization for displayed/saved grids: None | 'max' | 'minmax' | 'zscore'
#     normalize: Optional[str] = None,

#     # WHAT to display in the notebook: 'animation' | 'snapshot' | 'mean' | 'union'
#     display: str = "animation",

#     # optional RRT tree overlay
#     draw_rrt: bool = False,
#     Va: Optional[Iterable[Iterable[float]]] = None,
#     Ea: Optional[Iterable[Iterable[int]]] = None,
#     Vb: Optional[Iterable[Iterable[float]]] = None,
#     Eb: Optional[Iterable[Iterable[int]]] = None,
#     run_name: str = "stp-rrt*"
# ):
#     """
#     Renders Net PoD either as an animation or as a requested static heatmap (snapshot/mean/union).
#     Saves PNGs to image/heatmap/. Only the selected `display` figure is shown.

#     Returns:
#       - If display='animation': (fig_anim, anim, payload)
#       - Else (snapshot/mean/union): (fig_static, payload)
#     payload may include keys: 'snapshot', 'mean', 'union' with arrays.
#     """
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation, writers
#     from matplotlib.patches import Polygon as PolyPatch, Wedge, FancyArrow

#     # ---------- helpers ----------
#     def _apply_norm(G: np.ndarray, mode: Optional[str]) -> np.ndarray:
#         if mode is None:
#             return G
#         if mode == "max":
#             m = float(np.nanmax(G));  return G / m if m > 0 else G
#         if mode == "minmax":
#             gmin = float(np.nanmin(G)); gmax = float(np.nanmax(G))
#             return (G - gmin) / (gmax - gmin) if gmax > gmin else np.zeros_like(G)
#         if mode == "zscore":
#             mu = float(np.nanmean(G)); sd = float(np.nanstd(G))
#             return (G - mu) / sd if sd > 0 else G - mu
#         raise ValueError("normalize must be one of None, 'max', 'minmax', 'zscore'")

#     def _save_png(fig, basename: str):
#         out_dir = os.path.join("image", "heatmap")
#         os.makedirs(out_dir, exist_ok=True)
#         path = os.path.join(out_dir, basename)
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         return path

#     if pod_fn is None:
#         pod_fn = lambda rr, max_r: pod_exponential(rr, r0=0.5*max_r)

#     # ---------- map ----------
#     st = map_in['st']
#     size = np.array(st['size'], dtype=float)     # [xmin, xmax, ymin, ymax]
#     x_map = float(size[1] - size[0])
#     y_map = float(size[3] - size[2])
#     buildings = []
#     for i in range(int(st['n'])):
#         poly = np.array(st[str(i)], dtype=float)
#         if size[0] != 0.0 or size[2] != 0.0:
#             poly[:,0] -= size[0]; poly[:,1] -= size[2]
#         buildings.append([tuple(p) for p in poly])

#     # ---------- cameras ----------
#     n = int(cam_dict['n'])
#     cam_x = np.array(cam_dict['x'], dtype=float)
#     cam_y = np.array(cam_dict['y'], dtype=float)
#     spec = cam_dict['spec']
#     init_angle = [float(a) for a in spec['init_angle']]
#     bound_arr  = np.array(spec['bound'], dtype=float)   # (n,2): [+max, -max]
#     fov_ang    = float(spec['fov'][0])                  # radians (half-angle)
#     fov_rng    = float(spec['fov'][1])                  # max range
#     num_frames = int(spec['cam_time'][0])
#     dt         = float(spec['cam_time'][1])
#     panspeed   = [float(w) for w in spec['panspeed']]
#     if size[0] != 0.0 or size[2] != 0.0:
#         cam_x = cam_x - size[0]; cam_y = cam_y - size[2]

#     cameras = []
#     for i in range(n):
#         cameras.append({
#             "x": cam_x[i], "y": cam_y[i],
#             "theta0": init_angle[i],
#             "bound_plus": bound_arr[i,0], "bound_minus": bound_arr[i,1],
#             "fov_half": fov_ang, "max_range": fov_rng, "panspeed": panspeed[i]
#         })

#     # ---------- path ----------
#     path_arr = np.array(path_array, dtype=float)
#     if path_arr.ndim != 2 or path_arr.shape[1] not in (2,3):
#         raise ValueError("path_array must have shape (N,3) [x,y,t] or (N,2) [x,y].")
#     if path_arr.shape[1] == 2:
#         # synthesize time with camera dt
#         t_col = np.arange(path_arr.shape[0], dtype=float) * dt
#         path_arr = np.c_[path_arr, t_col]
#     if size[0] != 0.0 or size[2] != 0.0:
#         path_arr[:,0] -= size[0]; path_arr[:,1] -= size[2]
#     # ensure time monotone
#     order = np.argsort(path_arr[:,2])
#     path_arr = path_arr[order]
#     px, py, pt = path_arr[:,0], path_arr[:,1], path_arr[:,2]
#     t0, t_end = float(pt[0]), float(pt[-1])

#     # ---------- timelines ----------
#     frames_path = int(np.ceil((t_end - t0) / dt)) + 1
#     frames_anim = max(num_frames, frames_path)     # for animation/snapshot resample
#     frames_mean = frames_path                      # for mean/union over run only

#     t_frames = np.arange(frames_anim) * dt + t0
#     target_x = np.interp(t_frames, pt, px)
#     target_y = np.interp(t_frames, pt, py)

#     # ---------- grid ----------
#     nx, ny = int(grid_resolution[0]), int(grid_resolution[1])
#     gx = np.linspace(0.0, x_map, nx); gy = np.linspace(0.0, y_map, ny)
#     XX, YY = np.meshgrid(gx, gy)

#     # ---------- edges & spatial index ----------
#     all_edges = []
#     for poly in buildings:
#         for e1, e2 in polygon_edges(poly):
#             all_edges.append((e1, e2))
#     edge_index = EdgeIndex(all_edges, (x_map, y_map), buckets_per_axis=los_buckets_per_axis)

#     # ---------- per-camera range grids ----------
#     range_grids = [np.hypot(XX - c["x"], YY - c["y"]) for c in cameras]

#     # ---------- pan model ----------
#     def camera_angle_at_frame(cam, k):
#         theta0 = cam["theta0"]; w = cam["panspeed"]
#         lo = theta0 + cam["bound_minus"]; hi = theta0 + cam["bound_plus"]
#         if lo > hi: lo, hi = hi, lo
#         if abs(w) < 1e-12 or hi == lo: return theta0
#         span = hi - lo; raw = theta0 + w*(k*dt); period = 2.0*span
#         off = (raw - lo) % period
#         return (lo + off) if (off <= span) else (hi - (off - span))

#     # ---------- LOS masks (correct: only range-gate, no FOV pruning) ----------
#     los_masks = []
#     for cam, rr in zip(cameras, range_grids):
#         candidate = (rr <= cam["max_range"])
#         if los_blocks:
#             los = np.zeros_like(rr, dtype=bool)
#             ii, jj = np.where(candidate)
#             spt = (cam["x"], cam["y"])
#             for a, b in zip(ii, jj):
#                 dpt = (float(XX[a, b]), float(YY[a, b]))
#                 if los_is_clear_indexed(spt, dpt, edge_index, all_edges):
#                     los[a, b] = True
#         else:
#             los = np.ones_like(rr, dtype=bool)
#         los_masks.append(los)

#     # ---------- per-frame network PoD ----------
#     def network_pod_grid(k):
#         prod_not = np.ones_like(XX, dtype=float)
#         for cam, rr, los in zip(cameras, range_grids, los_masks):
#             facing  = camera_angle_at_frame(cam, k)
#             fov_ok  = in_fov(XX - cam["x"], YY - cam["y"], facing, cam["fov_half"])
#             range_ok = rr <= cam["max_range"]
#             mask    = range_ok & fov_ok & los
#             P       = pod_fn(rr, cam["max_range"])
#             P       = np.where(mask, P, 0.0)
#             prod_not *= (1.0 - P)
#         return 1.0 - prod_not

#     # ---------- compute requested analytics ----------
#     payload = {}

#     # snapshot (if asked or needed for display)
#     if snapshot_t is not None or display == "snapshot":
#         t_snap = float(np.clip(snapshot_t if snapshot_t is not None else t0, t0, t_end))
#         k_snap = int(round((t_snap - t0) / dt))
#         k_snap = int(np.clip(k_snap, 0, frames_anim - 1))
#         P_snap = _apply_norm(network_pod_grid(k_snap), normalize)
#         payload["snapshot"] = {"t": k_snap*dt + t0, "k": k_snap, "P": P_snap, "X": XX, "Y": YY}

#     # time-average (over run duration only)
#     if compute_mean or display == "mean":
#         mean_grid = np.zeros_like(XX, dtype=float)
#         for k in range(frames_mean):
#             mean_grid += network_pod_grid(k)
#         mean_grid /= float(frames_mean)
#         mean_grid = _apply_norm(mean_grid, normalize)
#         payload["mean"] = {"P_mean": mean_grid, "X": XX, "Y": YY}

#     # temporal union (over run duration only)
#     if compute_union or display == "union":
#         prod_not = np.ones_like(XX, dtype=float)
#         for k in range(frames_mean):
#             prod_not *= (1.0 - network_pod_grid(k))
#         union_grid = 1.0 - prod_not
#         union_grid = _apply_norm(union_grid, normalize)
#         payload["union"] = {"P_union": union_grid, "X": XX, "Y": YY}

#     # ---------- decorate helper (no grid, white dashed solver path) ----------
#     def _decorate_axes(ax, title_suffix: str):
#         # buildings
#         for poly in buildings:
#             ax.add_patch(PolyPatch(poly, closed=True, fill=True, alpha=0.25, edgecolor="black"))
#         # sensors
#         ax.scatter([c["x"] for c in cameras], [c["y"] for c in cameras], marker="^", s=60)
#         # optional trees
#         if draw_rrt:
#             _maybe_draw_rrt(ax, Va, Ea, "tab:blue")
#             _maybe_draw_rrt(ax, Vb, Eb, "tab:green")
#         # full STP-RRT* path (white dashed)
#         ax.plot(px, py, "--", color="white", lw=1.5, alpha=0.8)
#         ax.set_xlim(0, x_map); ax.set_ylim(0, y_map)
#         ax.set_aspect('equal', adjustable='box')
#         ax.set_xlabel("x"); ax.set_ylabel("y")
#         # ax.set_title(f"Net PoD {title_suffix} ({run_name})")

#     # ---------- build exactly ONE figure to display ----------
#     # vmin/vmax for normalized [0,1] modes
#     vmin = 0.0 if normalize in (None, "max", "minmax") else None
#     vmax = 1.0 if normalize in (None, "max", "minmax") else None
#     cbar_suffix = "" if normalize is None else f" [{normalize}]"

#     if display == "animation":
#         # Build animation only in this mode
#         fig, ax = plt.subplots(figsize=(7, 7), dpi=250)
#         im = ax.imshow(np.zeros_like(XX), origin="lower",
#                        extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax,
#                        interpolation="nearest")
#         cbar = plt.colorbar(im, ax=ax); cbar.set_label("P(detect)" + cbar_suffix)

#         # initial wedges + heading
#         fov_wedges = []
#         for c in cameras:
#             theta_deg = math.degrees(c["theta0"])
#             wedge = Wedge(center=(c["x"], c["y"]),
#                           r=c["max_range"],
#                           theta1=theta_deg - math.degrees(c["fov_half"]),
#                           theta2=theta_deg + math.degrees(c["fov_half"]),
#                           alpha=0.08)
#             ax.add_patch(wedge); fov_wedges.append(wedge)
#             ax.add_patch(FancyArrow(c["x"], c["y"],
#                                     2.0*math.cos(math.radians(theta_deg)),
#                                     2.0*math.sin(math.radians(theta_deg)),
#                                     width=0.25, length_includes_head=True, alpha=0.6))

#         # background (no grid): buildings/sensors + white dashed full path
#         _decorate_axes(ax, "")

#         # moving target marker along resampled path
#         target_dot, = ax.plot([], [], "ro", ms=5)

#         def init():
#             im.set_data(np.zeros_like(XX))
#             target_dot.set_data([], [])
#             return [im, target_dot] + fov_wedges

#         def update(k):
#             Pnet = _apply_norm(network_pod_grid(k), normalize)
#             im.set_data(Pnet)
#             for c, w in zip(cameras, fov_wedges):
#                 theta_deg = math.degrees(camera_angle_at_frame(c, k))
#                 w.set_theta1(theta_deg - math.degrees(c["fov_half"]))
#                 w.set_theta2(theta_deg + math.degrees(c["fov_half"]))
#             kk = min(k, len(target_x)-1)
#             target_dot.set_data(target_x[kk], target_y[kk])
#             # ax.set_title(f"Net PoD ({run_name})  |  t={k*dt + t0:.2f}")
#             return [im, target_dot] + fov_wedges

#         anim = FuncAnimation(fig, update, frames=frames_anim, init_func=init, blit=True, interval=50)
#         # save first frame preview
#         _ = _save_png(fig, f"{run_name}_animation_frame0.png")
#         return fig, anim, payload

#     # Otherwise build the chosen static figure:
#     fig_s, ax_s = plt.subplots(figsize=(7, 7), dpi=250)

#     if display == "snapshot":
#         P = payload["snapshot"]["P"]
#         im = ax_s.imshow(P, origin="lower", extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax, interpolation="nearest")
#         cbar = plt.colorbar(im, ax=ax_s); cbar.set_label("Pd")
#         _decorate_axes(ax_s, f"snapshot t={payload['snapshot']['t']:.2f}")
#         _ = _save_png(fig_s, f"{run_name}_snapshot.png")
#         return fig_s, payload

#     if display == "mean":
#         P = payload["mean"]["P_mean"]
#         im = ax_s.imshow(P, origin="lower", extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax, interpolation="nearest")
#         cbar = plt.colorbar(im, ax=ax_s); cbar.set_label("Pd")
#         _decorate_axes(ax_s, "time-average")
#         _ = _save_png(fig_s, f"{run_name}_mean.png")
#         return fig_s, payload

#     if display == "union":
#         P = payload["union"]["P_union"]
#         im = ax_s.imshow(P, origin="lower", extent=[0, x_map, 0, y_map], vmin=vmin, vmax=vmax, interpolation="nearest")
#         cbar = plt.colorbar(im, ax=ax_s); cbar.set_label("Pd")
#         _decorate_axes(ax_s, "temporal-union")
#         _ = _save_png(fig_s, f"{run_name}_union.png")
#         return fig_s, payload

#     raise ValueError("display must be one of 'animation', 'snapshot', 'mean', 'union'")





# # ----------------- Helpers -----------------

# def _maybe_draw_rrt(ax, V: Optional[Iterable[Iterable[float]]], E: Optional[Iterable[Iterable[int]]],
#                     color: str, label: Optional[str] = None):
#     if V is None or E is None:
#         return
#     V = np.array(V, dtype=float)
#     if V.shape[1] >= 2:
#         x, y = V[:,0], V[:,1]
#     else:
#         return
#     # scatter vertices (light)
#     ax.scatter(x, y, s=4, alpha=0.3, color=color, zorder=2)
#     # draw edges
#     for e in E:
#         try:
#             i, j = int(e[0]), int(e[1])
#             ax.plot([x[i], x[j]], [y[i], y[j]], lw=0.5, alpha=0.3, color=color, zorder=1)
#         except Exception:
#             continue
