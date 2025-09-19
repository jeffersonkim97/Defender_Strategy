"""
heatmap_animation.py

Takes:
  - map_in, cam_dict  (as produced by map_generator_matching_user.generate_map)
  - path JSON file with your format:
        {"RRTstar": {"some_run_key": [[x, y, t], ...], ...}}
    choose run via run_key, else first key is picked deterministically.

Outputs:
  - Time-varying detection heatmap animation, with fast LOS:
      * Range+FOV gating before ray tracing
      * Uniform-grid spatial index for building edges

USAGE (from your code):
    from map_generator_matching_user import generate_map
    from heatmap_animation import generate_heatmap

    map_in, cam_dict = generate_map(map_size=(30,30), num_buildings=1, num_sensors=2, ...)
    fig, anim = generate_heatmap(
        map_in=map_in,
        cam_dict=cam_dict,
        path_json_path="2D_Comparison_Path.json",   # your file of [x,y,t]
        run_key="21",                                # or None to auto-pick
        grid_resolution=(120, 120),
        los_blocks=True,
        los_buckets_per_axis=32,
        save_path=None                               # e.g., "out.mp4" or "out.gif"
    )
"""

from __future__ import annotations
import json
import math
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patches import Polygon as PolyPatch, Wedge, FancyArrow

# ----------------- Path I/O -----------------

def load_path_json(path: str, run_key: Optional[str] = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("RRTstar", {})
    if not runs:
        raise ValueError("Path JSON missing 'RRTstar' or empty.")
    if run_key is None:
        # deterministic choice: smallest key by (len,key) to keep stable
        run_key = sorted(runs.keys(), key=lambda k: (len(k), k))[0]
    arr = np.array(runs[run_key], dtype=float)  # shape (N,3)
    return arr[:,0], arr[:,1], arr[:,2], run_key


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
    return np.exp(- (range_grid / r0) ** 2)


# ----------------- Main player -----------------

def generate_heatmap(
    map_in: Dict,
    cam_dict: Dict,
    path_json_path: str,
    run_key: Optional[str] = None,
    los_blocks: bool = True,
    los_buckets_per_axis: int = 32,
    grid_resolution: Tuple[int, int] = (100, 100),
    save_path: Optional[str] = None,
    pod_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
):
    """
    Render time-varying detection heatmap.

    Args:
        map_in, cam_dict: as returned by map_generator_matching_user.generate_map
        path_json_path: your file with RRTstar → run_key → [[x,y,t], ...]
        run_key: which run to play (None = auto-pick)
        los_blocks: enable LOS occlusion
        los_buckets_per_axis: granularity of edge spatial index
        grid_resolution: (#cells_x, #cells_y)
        save_path: optional .mp4 or .gif path
        pod_fn: optional PoD model; default exp(-(r/(0.5*max_range))^2)
    """
    if pod_fn is None:
        pod_fn = lambda rr, max_r: pod_exponential(rr, r0=0.5*max_r)

    # --- Map bounds & buildings from your 'st' block ---
    st = map_in['st']
    size = np.array(st['size']).astype(float)  # [xmin, xmax, ymin, ymax]
    x_map = float(size[1] - size[0])
    y_map = float(size[3] - size[2])

    buildings = []
    for i in range(int(st['n'])):
        poly = np.array(st[str(i)], dtype=float)
        # shift to [0,x_map]x[0,y_map] frame if size has nonzero mins
        if size[0] != 0.0 or size[2] != 0.0:
            poly[:,0] -= size[0]
            poly[:,1] -= size[2]
        buildings.append([tuple(p) for p in poly])

    # --- Camera spec from your cam_dict ---
    n = int(cam_dict['n'])
    cam_x = np.array(cam_dict['x'], dtype=float)
    cam_y = np.array(cam_dict['y'], dtype=float)
    spec = cam_dict['spec']
    init_angle = [float(a) for a in spec['init_angle']]
    bound_arr = np.array(spec['bound'], dtype=float)  # shape (n, 2): [ +max, -max ]
    fov_ang = float(spec['fov'][0])
    fov_rng = float(spec['fov'][1])
    num_frames = int(spec['cam_time'][0])
    dt = float(spec['cam_time'][1])
    panspeed = [float(w) for w in spec['panspeed']]

    # If original coords used nonzero mins, shift cameras as well
    if size[0] != 0.0 or size[2] != 0.0:
        cam_x = cam_x - size[0]
        cam_y = cam_y - size[2]

    cameras = []
    for i in range(n):
        cameras.append({
            "x": cam_x[i],
            "y": cam_y[i],
            "theta0": init_angle[i],
            "bound_plus": bound_arr[i,0],
            "bound_minus": bound_arr[i,1],
            "fov_half": fov_ang,
            "max_range": fov_rng,
            "panspeed": panspeed[i]
        })

    # --- Load and resample the path onto the camera timeline ---
    px, py, pt, chosen_key = load_path_json(path_json_path, run_key=run_key)
    frames = num_frames
    t_frames = np.arange(frames) * dt
    target_x = np.interp(t_frames, pt, px)
    target_y = np.interp(t_frames, pt, py)

    # --- Grid ---
    nx, ny = int(grid_resolution[0]), int(grid_resolution[1])
    gx = np.linspace(0.0, x_map, nx)
    gy = np.linspace(0.0, y_map, ny)
    XX, YY = np.meshgrid(gx, gy)

    # --- Global edge list + spatial index for fast LOS ---
    all_edges = []
    for poly in buildings:
        for e1, e2 in polygon_edges(poly):
            all_edges.append((e1, e2))
    edge_index = EdgeIndex(all_edges, (x_map, y_map), buckets_per_axis=los_buckets_per_axis)

    # --- Precompute range grids ---
    range_grids = []
    for cam in cameras:
        dx = XX - cam["x"]
        dy = YY - cam["y"]
        rr = np.hypot(dx, dy)
        range_grids.append(rr)

    # --- Camera motion (bounce within [theta0+minus, theta0+plus]) ---
    def camera_angle_at_frame(cam, k):
        theta0 = cam["theta0"]
        w = cam["panspeed"]
        lo = theta0 + cam["bound_minus"]
        hi = theta0 + cam["bound_plus"]
        if lo > hi:
            lo, hi = hi, lo
        if abs(w) < 1e-12 or hi == lo:
            return theta0
        span = hi - lo
        raw = theta0 + w*(k*dt)
        period = 2.0*span
        off = (raw - lo) % period
        return (lo + off) if (off <= span) else (hi - (off - span))

    # --- Precompute static LOS masks with cheap gating (using initial angle only for pruning) ---
    los_masks = []
    for cam, rr in zip(cameras, range_grids):
        range_ok = rr <= cam["max_range"]
        fov_ok_init = in_fov(XX - cam["x"], YY - cam["y"], cam["theta0"], cam["fov_half"])
        candidate = range_ok & fov_ok_init

        if los_blocks:
            los = np.zeros_like(rr, dtype=bool)
            ii, jj = np.where(candidate)
            spt = (cam["x"], cam["y"])
            for a, b in zip(ii, jj):
                dpt = (float(XX[a, b]), float(YY[a, b]))
                if los_is_clear_indexed(spt, dpt, edge_index, all_edges):
                    los[a, b] = True
        else:
            los = np.ones_like(rr, dtype=bool)
        los_masks.append(los)

    # --- Network PoD per frame: 1 - prod(1 - P_i) ---
    def network_pod_grid(k):
        prod_not = np.ones_like(XX, dtype=float)
        for cam, rr, los in zip(cameras, range_grids, los_masks):
            facing = camera_angle_at_frame(cam, k)
            fov_ok = in_fov(XX - cam["x"], YY - cam["y"], facing, cam["fov_half"])
            range_ok = rr <= cam["max_range"]
            mask = range_ok & fov_ok & los

            P = pod_fn(rr, cam["max_range"])
            P = np.where(mask, P, 0.0)
            prod_not *= (1.0 - P)
        return 1.0 - prod_not

    # --- Plot & animate (matplotlib) ---
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(np.zeros_like(XX), origin="lower",
                   extent=[0, x_map, 0, y_map],
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("P(detect)")

    # Buildings
    for poly in buildings:
        ax.add_patch(PolyPatch(poly, closed=True, fill=True, alpha=0.25, edgecolor="black"))

    # Sensors & initial wedges
    ax.scatter([c["x"] for c in cameras], [c["y"] for c in cameras], marker="^", s=60)
    fov_wedges = []
    for c in cameras:
        theta_deg = math.degrees(c["theta0"])
        wedge = Wedge(center=(c["x"], c["y"]),
                      r=c["max_range"],
                      theta1=theta_deg - math.degrees(c["fov_half"]),
                      theta2=theta_deg + math.degrees(c["fov_half"]),
                      alpha=0.08)
        ax.add_patch(wedge); fov_wedges.append(wedge)
        ax.add_patch(FancyArrow(c["x"], c["y"],
                                2.0*math.cos(math.radians(theta_deg)),
                                2.0*math.sin(math.radians(theta_deg)),
                                width=0.25, length_includes_head=True, alpha=0.6))

    # Path and target marker
    ax.plot(target_x, target_y, "r--", lw=1, alpha=0.6)
    target_dot, = ax.plot([], [], "ro", ms=5)

    ax.set_xlim(0, x_map); ax.set_ylim(0, y_map)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(f"Net PoD (run='{chosen_key}')")

    def init():
        im.set_data(np.zeros_like(XX))
        target_dot.set_data([], [])
        return [im, target_dot] + fov_wedges

    def update(k):
        Pnet = network_pod_grid(k)
        im.set_data(Pnet)
        for c, w in zip(cameras, fov_wedges):
            theta_deg = math.degrees(camera_angle_at_frame(c, k))
            w.set_theta1(theta_deg - math.degrees(c["fov_half"]))
            w.set_theta2(theta_deg + math.degrees(c["fov_half"]))
        target_dot.set_data(target_x[k], target_y[k])
        ax.set_title(f"Net PoD (run='{chosen_key}')  |  t={k*dt:.2f}")
        return [im, target_dot] + fov_wedges

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

    # Optional save
    if save_path:
        ext = save_path.lower().split(".")[-1]
        if ext == "mp4":
            try:
                Writer = writers["ffmpeg"]
                writer = Writer(fps=max(1, int(1.0 / max(0.05, dt))),
                                metadata=dict(artist="net-pod"),
                                bitrate=2000)
                anim.save(save_path, writer=writer, dpi=150)
            except Exception as exc:
                print("FFMPEG not available or save failed:", exc)
        elif ext == "gif":
            try:
                anim.save(save_path, writer="pillow", dpi=150)
            except Exception as exc:
                print("GIF save failed:", exc)
        else:
            print("Unknown extension; not saved. Use .mp4 or .gif.")

    return fig, anim


# ------------------- CLI test -------------------
if __name__ == "__main__":
    # Tiny self-check using a synthetic one-building map
    import numpy as np

    # Build a quick single-rect map like your example
    map_size = [30, 30]
    static_obstacle_pos = [[[10, -5], 10, 25]]  # [anchor(x,y), width, height]
    poly = np.array([
        (static_obstacle_pos[0][0][0],                          static_obstacle_pos[0][0][1]),
        (static_obstacle_pos[0][0][0],                          static_obstacle_pos[0][0][1]+static_obstacle_pos[0][2]),
        (static_obstacle_pos[0][0][0]+static_obstacle_pos[0][1],static_obstacle_pos[0][0][1]+static_obstacle_pos[0][2]),
        (static_obstacle_pos[0][0][0]+static_obstacle_pos[0][1],static_obstacle_pos[0][0][1])
    ], dtype=float)

    map_in = {'st': {'size': np.array([0, map_size[0], 0, map_size[1]], dtype=float),
                     'n': 1,
                     '0': poly},
              'n': 200,
              'ncam': 2}

    cam_dict = {
        'n': 2,
        'x': np.array([10.0, 20.0]),
        'y': np.array([20.0, 20.0]),
        'spec': {
            'init_angle': [math.pi, 0.0],
            'bound': np.array([[math.radians(90), math.radians(-90)],
                               [math.radians(90), math.radians(-90)]]),
            'fov': [math.radians(25), 12.0],
            'cam_time': [200, 0.25],
            'panspeed': [math.radians(-2.5), math.radians(2.5)]
        }
    }

    # This expects a real path JSON; change path below or just run your own caller.
    # from this CLI test, we will skip calling generate_heatmap.
    print("CLI test built a map & cam_dict; import this module and call generate_heatmap with your path JSON.")
