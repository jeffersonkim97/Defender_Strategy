"""
animate_ad_game.py
==================
Animates the alternating attacker-defender game convergence (single panel).

The animation starts from the true initial sensor positions (from cam_dict)
and morphs through S_history toward the converged optimal placement.
Sensor FOV wedges are fixed at their t=0 facing angles — they do not rotate.
The initial RRT* path is oversampled to match the optimizer's vertex count
(A_history[0].shape[1]) so morphing between iterations is smooth.

Usage
-----
from animate_ad_game import animate_ad_game

# Build S_init from cam_dict initial positions — same structure as S_star
cam_direc_x = np.array(cam_dict['directional']['x'], dtype=float)
cam_direc_y = np.array(cam_dict['directional']['y'], dtype=float)
cam_omni_x  = np.array(cam_dict['omnidirectional']['x'], dtype=float)
cam_omni_y  = np.array(cam_dict['omnidirectional']['y'], dtype=float)
S_init = np.vstack([
    np.hstack([cam_direc_x, cam_omni_x]),   # row 0: x positions of all sensors
    np.hstack([cam_direc_y, cam_omni_y]),   # row 1: y positions of all sensors
])  # shape (2, M)

ani = animate_ad_game(
    map_in           = map_in,
    cam_dict         = cam_dict,
    path             = path,          # initial RRT* path: list of [x, y, t]
    A_history        = A_history,     # list of (3, N) arrays from iteration loop
    S_history        = S_history,     # list of (2, M) arrays from iteration loop
    S_init           = S_init,        # (2, M) true initial sensor positions
    x0               = x0,
    xf               = xf,
    num_direc_sensor = num_direc_sensor,
    num_omni_sensor  = num_omni_sensor,
    duration_s       = 10,
    fps              = 30,
    save_path        = None,
)
plt.show()
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as PolyPatch, Wedge


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bounce_angle(init_angle, bound_max, bound_min, panspeed, t):
    """Replicate map_visualization._bounce_angle."""
    span = abs(bound_max - bound_min)
    if panspeed == 0 or span == 0:
        return init_angle
    phase = (t * abs(panspeed)) % (2 * span)
    if phase < span:
        return init_angle + math.copysign(phase, panspeed)
    else:
        return init_angle + math.copysign(2 * span - phase, -panspeed)


def _resample_path_xy(path_xy, n):
    """Resample a (K, 2) xy array to exactly n points via linear interpolation."""
    path_xy = np.asarray(path_xy, dtype=float)
    t_orig = np.linspace(0.0, 1.0, len(path_xy))
    t_new  = np.linspace(0.0, 1.0, n)
    x = np.interp(t_new, t_orig, path_xy[:, 0])
    y = np.interp(t_new, t_orig, path_xy[:, 1])
    return np.column_stack([x, y])


def _ease(t):
    """Smooth-step easing."""
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_ad_game(
    map_in,
    cam_dict,
    path,
    A_history,
    S_history,
    S_init,
    x0,
    xf,
    num_direc_sensor,
    num_omni_sensor,
    duration_s=10,
    fps=30,
    save_path=None,
    marker_size_for_sensor=100,
    sensor_alpha=0.5,
    figsize=(8, 8),
    dpi=150,
):
    """
    Parameters
    ----------
    map_in              : dict   from generate_map()
    cam_dict            : dict   from generate_map()
    path                : list   initial RRT* path — list of [x, y, t]
    A_history           : list   iteration snapshots; each entry shape (3, N_a)
                                 rows = [x, y, t].
    S_history           : list   iteration snapshots; each entry shape (2, M)
                                 rows = [x, y] for all M sensors.
    S_init              : ndarray  shape (2, M) — TRUE initial sensor positions
                                 built directly from cam_dict:
                                   np.vstack([
                                       np.hstack([cam_dict['directional']['x'],
                                                  cam_dict['omnidirectional']['x']]),
                                       np.hstack([cam_dict['directional']['y'],
                                                  cam_dict['omnidirectional']['y']]),
                                   ])
    x0, xf             : list/array  start and goal  [x, y, ...]
    num_direc_sensor    : int
    num_omni_sensor     : int
    duration_s          : float  (default 10)
    fps                 : int    (default 30)
    save_path           : str|None  '.mp4' (needs ffmpeg) or '.gif' (needs Pillow)
    marker_size_for_sensor : int  (default 100)
    sensor_alpha        : float  FOV wedge alpha (default 0.5)
    figsize, dpi        : passed to plt.subplots

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """

    # ------------------------------------------------------------------ #
    #  1. Unpack static map geometry                                       #
    # ------------------------------------------------------------------ #
    st   = map_in["st"]
    size = np.array(st["size"], dtype=float)
    xmin, xmax, ymin, ymax = size
    buildings = [np.array(st[str(i)], dtype=float) for i in range(st["n"])]

    # ------------------------------------------------------------------ #
    #  2. Unpack camera / sensor data                                      #
    # ------------------------------------------------------------------ #
    spec        = cam_dict["directional"]["spec"]
    fov_half    = float(spec["fov"][0])
    fov_range   = float(spec["fov"][1])
    init_angles = np.array(spec["init_angle"], dtype=float)
    bound_arr   = np.array(spec["bound"],      dtype=float)   # (n_direc, 2)
    panspeeds   = np.array(spec["panspeed"],   dtype=float)

    has_omni = cam_dict.get("n_omni", 0) > 0
    if has_omni:
        fov_range_omni = float(cam_dict["omnidirectional"]["spec"]["fov"][1])

    # Wedge angles fixed at t=0 — do not rotate during animation
    facings_fixed = np.array([
        _bounce_angle(init_angles[i], bound_arr[i, 0], bound_arr[i, 1], panspeeds[i], 0)
        for i in range(num_direc_sensor)
    ], dtype=float)

    # ------------------------------------------------------------------ #
    #  3. Prepare path data                                                #
    # ------------------------------------------------------------------ #
    # Oversample initial RRT* path to match optimizer vertex count
    N_pts = A_history[0].shape[1]   # N_attk + 1

    init_xy_raw = np.array([[p[0], p[1]] for p in path], dtype=float)
    init_xy = _resample_path_xy(init_xy_raw, N_pts)

    # Resample every A_history entry to N_pts
    A_resampled = []
    for A in A_history:
        xy_k = np.column_stack([A[0, :], A[1, :]])
        A_resampled.append(_resample_path_xy(xy_k, N_pts))

    # Prepend the oversampled initial path so frame 0 starts from RRT*
    A_full = [init_xy] + A_resampled   # length = n_iters + 1

    # ------------------------------------------------------------------ #
    #  4. Prepare sensor position sequence                                 #
    # ------------------------------------------------------------------ #
    # S_init  : (2, M) true cam_dict positions  — animation frame 0
    # S_history[k] : (2, M) after iteration k  — animation frames 1..n_iters
    #
    # Prepending S_init means the animation visibly travels from the real
    # initial placement all the way to the converged optimal placement.
    S_full = [np.asarray(S_init, dtype=float)] + \
             [np.asarray(S, dtype=float) for S in S_history]   # length = n_iters + 1

    n_steps  = len(A_full)   # = len(S_full)
    n_frames = int(duration_s * fps)

    def _get_frame_data(frame_idx):
        t_global   = frame_idx / max(n_frames - 1, 1)
        step_float = _ease(t_global) * (n_steps - 1)
        lo    = int(step_float)
        hi    = min(lo + 1, n_steps - 1)
        alpha = step_float - lo
        xy = A_full[lo] * (1 - alpha) + A_full[hi] * alpha
        S  = S_full[lo] * (1 - alpha) + S_full[hi] * alpha
        # Report iteration index: step 0 = "initial", step k = "iter k"
        iter_label = lo if lo > 0 else 0
        return xy, S, t_global, iter_label

    # ------------------------------------------------------------------ #
    #  5. Build the figure                                                 #
    # ------------------------------------------------------------------ #
    plt.rcParams.update({"font.size": 14})

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for poly in buildings:
        ax.add_patch(PolyPatch(
            poly, closed=True, facecolor="black", edgecolor="black", alpha=1.0
        ))
    ax.plot(x0[0], x0[1], "H", c="r", markersize=10)
    ax.plot(xf[0], xf[1], "H", c="r", markersize=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  linewidth=0.5, color="gray", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, color="gray", alpha=0.3)

    # ------------------------------------------------------------------ #
    #  6. Set up animated artists                                          #
    # ------------------------------------------------------------------ #

    # Directional sensor scatter + wedges (angles baked in at construction)
    scat_dir = ax.scatter(
        [], [], s=marker_size_for_sensor,
        c="green", marker="o", zorder=3, label=r"$S^*_{\mathrm{Directional}}$"
    )
    wedge_artists = []
    for i in range(num_direc_sensor):
        theta_deg = math.degrees(facings_fixed[i])
        w = Wedge(
            center=(0, 0), r=fov_range,
            theta1=theta_deg - math.degrees(fov_half),
            theta2=theta_deg + math.degrees(fov_half),
            facecolor="green", edgecolor=None,
            alpha=sensor_alpha, zorder=2
        )
        ax.add_patch(w)
        wedge_artists.append(w)

    # Omnidirectional sensor scatter + circles
    omni_circles = []
    if has_omni:
        scat_omni = ax.scatter(
            [], [], s=marker_size_for_sensor,
            c="orange", marker="D", zorder=3, label=r"$S^*_{\mathrm{Omni}}$"
        )
        for _ in range(num_omni_sensor):
            c = plt.Circle(
                (0, 0), fov_range_omni,
                color="orange", alpha=0.25, clip_on=True, zorder=0
            )
            ax.add_patch(c)
            omni_circles.append(c)

    # Attacker path — one Line2D per segment
    opt_lines = []
    for _ in range(N_pts - 1):
        ln, = ax.plot([], [], "-", color="brown", linewidth=1.5, zorder=10)
        opt_lines.append(ln)
    opt_lines[0].set_label(r"$A^*$")

    ax.legend(loc="lower right", borderaxespad=0.)
    title_artist = ax.set_title("Initial strategies")

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  7. Animation update                                                 #
    # ------------------------------------------------------------------ #
    def update(frame):
        xy, S, t_global, iter_idx = _get_frame_data(frame)

        # Directional sensors — move center only, angles stay fixed
        cx_cur = S[0, :num_direc_sensor]
        cy_cur = S[1, :num_direc_sensor]
        scat_dir.set_offsets(np.column_stack([cx_cur, cy_cur]))
        for i, w in enumerate(wedge_artists):
            w.set_center((cx_cur[i], cy_cur[i]))

        # Omnidirectional sensors
        if has_omni:
            ox_cur = S[0, num_direc_sensor:num_direc_sensor + num_omni_sensor]
            oy_cur = S[1, num_direc_sensor:num_direc_sensor + num_omni_sensor]
            scat_omni.set_offsets(np.column_stack([ox_cur, oy_cur]))
            for i, c in enumerate(omni_circles):
                c.set_center((ox_cur[i], oy_cur[i]))

        # Attacker path segments
        for pi, ln in enumerate(opt_lines):
            ln.set_data(
                [xy[pi, 0], xy[pi + 1, 0]],
                [xy[pi, 1], xy[pi + 1, 1]]
            )

        if iter_idx == 0:
            title_artist.set_text("Initial strategies")
        else:
            title_artist.set_text(f"Optimal strategies — iter {iter_idx}")

        artists = [scat_dir, title_artist] + wedge_artists + opt_lines
        if has_omni:
            artists += [scat_omni] + omni_circles
        return artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
    )

    # ------------------------------------------------------------------ #
    #  8. Save or return                                                   #
    # ------------------------------------------------------------------ #
    if save_path is not None:
        if save_path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        ani.save(save_path, writer=writer)
        print(f"Saved animation to: {save_path}")

    return ani
