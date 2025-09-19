# convex_map.py
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np


# ==========================================================
# --------------------- Geometry utils ---------------------
# ==========================================================

def convex_hull_monotone_chain(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(points)
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return lower[:-1] + upper[:-1]


def polygon_edges(poly: List[Tuple[float, float]]):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i + 1) % n]


def polygon_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    x = np.array([p[0] for p in poly], dtype=float)
    y = np.array([p[1] for p in poly], dtype=float)
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    a = x * y2 - x2 * y
    A = a.sum() / 2.0
    cx = ((x + x2) * a).sum() / (6.0 * A)
    cy = ((y + y2) * a).sum() / (6.0 * A)
    return float(cx), float(cy)


def outward_normal_angle(edge: Tuple[Tuple[float, float], Tuple[float, float]],
                         centroid: Tuple[float, float]) -> float:
    """
    Returns the outward normal direction (radians, in [-pi, pi)) for the given edge.
    """
    (x1, y1), (x2, y2) = edge
    ex, ey = x2 - x1, y2 - y1
    # Two normals
    nx1, ny1 = -ey, ex
    nx2, ny2 = ey, -ex
    # Midpoint of edge, vector to centroid
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    vx, vy = centroid[0] - mx, centroid[1] - my
    # Choose the one pointing AWAY from centroid
    if nx1 * vx + ny1 * vy < 0:
        ang = math.atan2(ny1, nx1)
    else:
        ang = math.atan2(ny2, nx2)
    # Wrap to [-pi, pi)
    return (ang + math.pi) % (2.0 * math.pi) - math.pi


# ==========================================================
# --------------- Poisson-disk (blue-noise) ----------------
# ==========================================================

def poisson_disk_2d(width: float, height: float, radius: float, k: int = 30, rng: Optional[random.Random] = None):
    """
    Bridson Poisson-disk sampling on [0,width] x [0,height].
    Guarantees each sample is >= radius apart. Returns list of (x, y).
    """
    if rng is None:
        rng = random.Random()

    cell = radius / math.sqrt(2.0)
    gw = int(math.ceil(width / cell))
    gh = int(math.ceil(height / cell))
    grid = [[-1 for _ in range(gw)] for _ in range(gh)]
    samples: List[Tuple[float, float]] = []
    active: List[int] = []

    def gc(p):
        return int(p[0] // cell), int(p[1] // cell)

    def in_bounds(p):
        return 0.0 <= p[0] <= width and 0.0 <= p[1] <= height

    def far_enough(p):
        gx, gy = gc(p)
        for yy in range(max(0, gy - 2), min(gh, gy + 3)):
            for xx in range(max(0, gx - 2), min(gw, gx + 3)):
                idx = grid[yy][xx]
                if idx != -1:
                    qx, qy = samples[idx]
                    if (p[0] - qx) ** 2 + (p[1] - qy) ** 2 < radius ** 2:
                        return False
        return True

    # seed
    p0 = (rng.uniform(0, width), rng.uniform(0, height))
    samples.append(p0)
    active.append(0)
    g0x, g0y = gc(p0)
    grid[g0y][g0x] = 0

    while active:
        i = rng.choice(active)
        base = samples[i]
        found = False
        for _ in range(k):
            r = radius * (1.0 + rng.random())
            a = 2.0 * math.pi * rng.random()
            cand = (base[0] + r * math.cos(a), base[1] + r * math.sin(a))
            if in_bounds(cand) and far_enough(cand):
                samples.append(cand)
                active.append(len(samples) - 1)
                gx, gy = gc(cand)
                grid[gy][gx] = len(samples) - 1
                found = True
                break
        if not found:
            active.remove(i)

    return samples


# ==========================================================
# ----------------- World / building gen -------------------
# ==========================================================

def random_convex_polygon(center_xy: Tuple[float, float],
                          min_radius: float,
                          max_radius: float,
                          num_vertices: int) -> List[Tuple[float, float]]:
    """Convex-ish random polygon by sampling angles & radii then hull."""
    cx, cy = center_xy
    angles = np.sort(np.random.rand(num_vertices) * 2.0 * math.pi)
    radii = np.random.uniform(min_radius, max_radius, size=num_vertices)
    pts = [(cx + radii[i] * math.cos(angles[i]),
            cy + radii[i] * math.sin(angles[i])) for i in range(num_vertices)]
    return convex_hull_monotone_chain(pts)


def generate_non_overlapping_buildings(num_buildings: int,
                                       map_size: Tuple[float, float],
                                       rng_seed: int = 0,
                                       min_size: float = 4.0,
                                       max_size: float = 12.0,
                                       bbox_margin: float = 2.0,
                                       min_center_sep: float = 12.0) -> List[List[Tuple[float, float]]]:
    """
    Place building centers using Poisson-disk sampling (even spread), then
    create random convex polygons around those centers. Reject AABB overlaps.
    """
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)
    x_map, y_map = float(map_size[0]), float(map_size[1])

    # Sample candidate centers within an inset box to avoid clipping at borders
    inset = 4.0
    w_eff, h_eff = max(0.0, x_map - 2 * inset), max(0.0, y_map - 2 * inset)
    candidates = poisson_disk_2d(w_eff, h_eff, radius=min_center_sep, rng=rng)
    candidates = [(c[0] + inset, c[1] + inset) for c in candidates]
    rng.shuffle(candidates)

    buildings: List[List[Tuple[float, float]]] = []
    bboxes: List[Tuple[float, float, float, float]] = []

    for cx, cy in candidates:
        if len(buildings) >= num_buildings:
            break
        n_verts = rng.randint(4, 8)
        poly = random_convex_polygon((cx, cy), min_size, max_size, n_verts)

        xs, ys = zip(*poly)
        bxmin, bymin, bxmax, bymax = min(xs), min(ys), max(xs), max(ys)

        # Keep in map
        if bxmin < 0 or bymin < 0 or bxmax > x_map or bymax > y_map:
            continue

        # Loose AABB separation
        ok = True
        for (oxmin, oymin, oxmax, oymax) in bboxes:
            separated = (bxmax + bbox_margin < oxmin or bxmin - bbox_margin > oxmax or
                         bymax + bbox_margin < oymin or bymin - bbox_margin > oymax)
            if not separated:
                ok = False
                break

        if ok:
            buildings.append(poly)
            bboxes.append((bxmin, bymin, bxmax, bymax))

    return buildings


# ==========================================================
# ----------------- Sensor placement / schema --------------
# ==========================================================

def mount_sensors_on_edges(num_sensors: int,
                           buildings: List[List[Tuple[float, float]]],
                           rng_seed: int = 1,
                           outward_offset: float = 1e-3,
                           balanced: bool = False) -> List[Dict]:
    """
    Place sensors on building edges.
    - init_angle = outward normal
    - sensor position nudged outward by `outward_offset` to avoid numerical "inside" issues
    - if balanced=True, distribute sensors round-robin across buildings
    """
    rng = random.Random(rng_seed)
    sensors: List[Dict] = []

    if balanced:
        edges_by_build = [list(polygon_edges(b)) for b in buildings]
        b = 0
        for _ in range(num_sensors):
            edges = edges_by_build[b]
            e1, e2 = edges[rng.randrange(len(edges))]
            t = rng.random()
            sx = e1[0] + t * (e2[0] - e1[0])
            sy = e1[1] + t * (e2[1] - e1[1])
            theta0 = outward_normal_angle((e1, e2), polygon_centroid(buildings[b]))
            sx += outward_offset * math.cos(theta0)
            sy += outward_offset * math.sin(theta0)
            sensors.append({"pos": (float(sx), float(sy)), "theta0": float(theta0), "building": b})
            b = (b + 1) % len(buildings)
        return sensors

    # Unbalanced (random buildings/edges)
    for _ in range(num_sensors):
        b_idx = rng.randrange(len(buildings))
        poly = buildings[b_idx]
        edges = list(polygon_edges(poly))
        e1, e2 = edges[rng.randrange(len(edges))]
        t = rng.random()
        sx = e1[0] + t * (e2[0] - e1[0])
        sy = e1[1] + t * (e2[1] - e1[1])
        theta0 = outward_normal_angle((e1, e2), polygon_centroid(poly))
        sx += outward_offset * math.cos(theta0)
        sy += outward_offset * math.sin(theta0)
        sensors.append({"pos": (float(sx), float(sy)), "theta0": float(theta0), "building": b_idx})

    return sensors


def build_cam_dict_schema(
    sensors: List[Dict],
    # FOV
    fov_half_rad: float,
    max_range: float,
    # Bounds (if auto_outward_bounds=False, we treat these as desired limits and then clamp)
    bound_plus: float,
    bound_minus: float,
    # Panning
    panspeed_range: Tuple[float, float],
    cam_period: int,
    cam_increment: float,
    rng_seed: int = 2,
    # --- NEW defaults ---
    auto_outward_bounds: bool = True,
    max_outward_pan_deg: float = 85.0
) -> Dict:
    """
    Build cam_dict in YOUR schema.

    Defaults (NEW):
      - init_angle is the outward normal (from sensors list).
      - By default, we set symmetric outward-only bounds: ±max_outward_pan_deg about the outward normal,
        so cameras never pan into their own wall.
      - If you pass auto_outward_bounds=False, we will take (bound_plus, bound_minus) as desired
        and clamp them to ±max_outward_pan_deg to still prevent pointing into the wall.

    Returns:
      cam_dict with:
        'n', 'x', 'y', and 'spec' with keys:
          init_angle (radians), bound [[+,-],...], fov [half_rad, range],
          cam_time [period, increment], panspeed (list)
    """
    rng = random.Random(rng_seed)

    xs, ys, init_angles = [], [], []
    for s in sensors:
        xs.append(float(s["pos"][0]))
        ys.append(float(s["pos"][1]))
        init_angles.append(float(s["theta0"]))

    # Compute bounds per sensor
    cap = math.radians(max_outward_pan_deg)

    bounds: List[List[float]] = []
    if auto_outward_bounds:
        # Force symmetric outward-only bounds for every sensor
        for _ in sensors:
            bounds.append([+cap, -cap])
    else:
        # Clamp requested bounds to never exceed ±cap
        req_plus = float(bound_plus)
        req_minus = float(bound_minus)
        hi = min(cap, max(0.0, req_plus))
        lo = max(-cap, min(0.0, req_minus))
        for _ in sensors:
            bounds.append([hi, lo])

    # Panspeeds
    panspeeds = [float(rng.uniform(*panspeed_range)) for _ in sensors]

    cam_dict = {
        "n": len(sensors),
        "x": np.array(xs, dtype=float),
        "y": np.array(ys, dtype=float),
        "spec": {
            "init_angle": init_angles,                  # radians
            "bound": np.array(bounds, dtype=float),     # per-sensor [ +, - ] (relative to init_angle)
            "fov": [float(fov_half_rad), float(max_range)],
            "cam_time": [int(cam_period), float(cam_increment)],
            "panspeed": panspeeds                       # rad / time
        }
    }
    return cam_dict


# ==========================================================
# --------------------- Public API -------------------------
# ==========================================================

def generate_map(map_size: Tuple[float, float] = (100.0, 100.0),
                 num_buildings: int = 8,
                 num_sensors: int = 16,
                 # FOV (radians, range)
                 fov_half_rad: float = math.radians(25.0),
                 max_range: float = 12.0,
                 # Legacy inputs for bounds (kept for compatibility; ignored if auto_outward_bounds=True)
                 bound_plus: float = math.pi / 2.0,
                 bound_minus: float = -math.pi / 2.0,
                 # Panning
                 panspeed_range: Tuple[float, float] = (math.radians(-2.5), math.radians(2.5)),
                 cam_period: int = 200,
                 cam_increment: float = 0.25,
                 # RNG & placement controls
                 rng_seed: int = 7,
                 balanced_sensors: bool = False,
                 min_center_sep: float = 12.0,
                 # NEW defaults controlling “outward only” behavior
                 auto_outward_bounds: bool = True,
                 max_outward_pan_deg: float = 85.0,
                 outward_offset: float = 1e-3) -> Tuple[Dict, Dict]:
    """
    Generate buildings + edge-mounted sensors and return (map_in, cam_dict) in YOUR schema.

    - Buildings: Poisson-disk centers (even spread), convex shapes, AABB-checked.
    - Sensors: placed on edges, init_angle = outward normal; nudged outward by 'outward_offset'.
    - Bounds default: cameras pan OUTWARD only, symmetric ±max_outward_pan_deg (85°) around outward normal.
      Set auto_outward_bounds=False to use your (bound_plus, bound_minus) but still clamp to ±max_outward_pan_deg.

    Returns:
      map_in:
        {
          'st': {
            'size': np.array([xmin, xmax, ymin, ymax]),
            'n': <#buildings>,
            '0': np.array([(x1,y1), ...]),
            '1': ...
          },
          'n': cam_period,            # like your example
          'ncam': cam_dict['n']
        }

      cam_dict: as documented above.
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # 1) Buildings
    buildings = generate_non_overlapping_buildings(
        num_buildings=num_buildings,
        map_size=map_size,
        rng_seed=rng_seed,
        min_size=4.0,
        max_size=12.0,
        bbox_margin=2.0,
        min_center_sep=min_center_sep
    )

    # 2) Sensors
    sensors = mount_sensors_on_edges(
        num_sensors=num_sensors,
        buildings=buildings,
        rng_seed=rng_seed + 1,
        outward_offset=outward_offset,
        balanced=balanced_sensors
    )

    # 3) map_in['st']
    xmin, ymin = 0.0, 0.0
    xmax, ymax = float(map_size[0]), float(map_size[1])
    st: Dict = {"size": np.array([xmin, xmax, ymin, ymax], dtype=float), "n": len(buildings)}
    for i, poly in enumerate(buildings):
        st[str(i)] = np.array(poly, dtype=float)

    # 4) cam_dict (init_angle outward; bounds default outward-only)
    cam_dict = build_cam_dict_schema(
        sensors=sensors,
        fov_half_rad=fov_half_rad,
        max_range=max_range,
        bound_plus=bound_plus,
        bound_minus=bound_minus,
        panspeed_range=panspeed_range,
        cam_period=cam_period,
        cam_increment=cam_increment,
        rng_seed=rng_seed + 2,
        auto_outward_bounds=auto_outward_bounds,
        max_outward_pan_deg=max_outward_pan_deg
    )

    # 5) map_in top-level fields (your pattern)
    map_in: Dict = {
        "st": st,
        "n": int(cam_period),
        "ncam": int(cam_dict["n"])
    }

    return map_in, cam_dict


# ==========================================================
# ------------------ Quick sanity demo ---------------------
# ==========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolyPatch, Wedge

    # Example usage
    m, cams = generate_map(
        map_size=(120, 80),
        num_buildings=10,
        num_sensors=20,
        fov_half_rad=np.deg2rad(25),
        max_range=14.0,
        # These are ignored by default since auto_outward_bounds=True,
        # but kept for compatibility if you set auto_outward_bounds=False:
        bound_plus=np.deg2rad(120),
        bound_minus=np.deg2rad(-120),
        panspeed_range=(np.deg2rad(-2.5), np.deg2rad(2.5)),
        cam_period=240,
        cam_increment=0.25,
        rng_seed=7,
        balanced_sensors=True,            # spread sensors across buildings
        min_center_sep=12.0,
        auto_outward_bounds=True,         # DEFAULT: outward-only pan
        max_outward_pan_deg=85.0,         # cap to avoid pointing into wall
        outward_offset=1e-3
    )

    # quick t=0 visualization
    st = m["st"]
    size = st["size"]
    cx = cams["x"]; cy = cams["y"]
    spec = cams["spec"]
    init_angles = np.array(spec["init_angle"], dtype=float)
    fov_half = float(spec["fov"][0]); fov_range = float(spec["fov"][1])

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(st["n"]):
        poly = st[str(i)]
        ax.add_patch(PolyPatch(poly, closed=True, facecolor="black", edgecolor="black", alpha=1.0))

    ax.scatter(cx, cy, s=36, c="green", marker="o", zorder=3)
    for i in range(len(cx)):
        theta_deg = math.degrees(init_angles[i])
        w = Wedge(center=(cx[i], cy[i]),
                  r=fov_range,
                  theta1=theta_deg - math.degrees(fov_half),
                  theta2=theta_deg + math.degrees(fov_half),
                  facecolor="green", edgecolor=None, alpha=0.5, zorder=2)
        ax.add_patch(w)

    ax.set_xlim(size[0], size[1]); ax.set_ylim(size[2], size[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("convex_map demo (t = 0)")
    plt.show()
