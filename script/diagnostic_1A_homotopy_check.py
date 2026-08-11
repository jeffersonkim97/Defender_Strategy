"""
Diagnostic 1-A: How often does the single-seed STP-RRT* + NLP attacker
best response leave a lower-cost (different-homotopy) trajectory on the
table, once evaluated against the SAME converged defender placement S*?

Method (see RAL_Review_Fix_0810.md, section 1-A):
  1. Run the existing bilevel scheme to convergence -> (A_star, S_star).
  2. Draw K alternate STP-RRT* seeds against the SAME map/defender.
  3. Refine each alternate with ONE call to optimize_attacker(S_star=S_star, ...)
     so it gets the same NLP polish as A_star (fair comparison).
  4. Compare C_AS(A_alt, S_star) against C_AS(A_star, S_star).
     Lower is better for the attacker (less detected).

Output: <timestamp>_1A_homotopy_diagnostic.json with per-trial records,
plus a printed summary (beat rate, margin distribution).
"""

import argparse
import json
import os
import random as rn
import time
from datetime import datetime

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
from scipy.interpolate import interp1d

from convex_map import generate_map
import convex_map
from Camera import Camera
from Dynamic import DynamicMap
from STP_RRTStar import STP_RRTStar as rrt
from json_seriablizable import make_json_serializable


# ----------------------------------------------------------------------
# Geometry / cost-model helpers (ported from
# attacker_defender_game_iterative_Monte_Carlo.ipynb, cell 905e00e1)
# ----------------------------------------------------------------------

def get_polygon_half_space_constraints(poly_vertices: np.ndarray):
    A_list, b_list = [], []
    N_verts = len(poly_vertices)
    if poly_vertices.ndim != 2:
        raise ValueError("Obstacle data is not a polygon vertex array.")
    centroid = convex_map.polygon_centroid([tuple(v) for v in poly_vertices])

    for i in range(N_verts):
        v1 = poly_vertices[i]
        v2 = poly_vertices[(i + 1) % N_verts]
        ex, ey = v2[0] - v1[0], v2[1] - v1[1]
        n1 = np.array([-ey, ex])
        norm_n1 = np.linalg.norm(n1)
        if norm_n1 > 1e-6:
            n1 = n1 / norm_n1
        mid_point = (v1 + v2) / 2.0
        vec_to_centroid = np.array(centroid) - mid_point
        a_i = n1 if np.dot(n1, vec_to_centroid) < 0 else -n1
        b_i = np.dot(a_i, v1)
        A_list.append(a_i)
        b_list.append(b_i)
    return np.array(A_list), np.array(b_list)


def build_obstacle_constraints(map_in):
    obstacle_constraints = []
    for i in range(map_in['st']['n']):
        poly_vertices = map_in['st'][str(i)]
        A, b = get_polygon_half_space_constraints(poly_vertices)
        obstacle_constraints.append({'A': A, 'b': b})
    return obstacle_constraints


def closest_edge(building, cam_xy, tol=None):
    P = np.asarray(building, dtype=float)
    x = np.asarray(cam_xy, dtype=float)
    M = P.shape[0]
    assert M >= 3

    Pn = np.vstack([P, P[0]])
    Ls = np.linalg.norm(Pn[1:] - Pn[:-1], axis=1)
    Lmean = float(np.mean(Ls))
    if tol is None:
        tol = 1e-3 * Lmean if Lmean > 0 else 1e-6

    best = None
    for i in range(M):
        p0_i, p1_i = Pn[i], Pn[i + 1]
        v_i = p1_i - p0_i
        L2 = float(v_i @ v_i)
        if L2 < 1e-16:
            continue
        t_raw = float((x - p0_i) @ v_i / L2)
        t_clamped = min(1.0, max(0.0, t_raw))
        proj_i = p0_i + t_clamped * v_i
        d2_i = float(np.sum((x - proj_i) ** 2))
        if (best is None) or (d2_i < best["d2"]):
            L = np.sqrt(L2)
            t_hat = v_i / L
            n_hat = np.array([-t_hat[1], t_hat[0]])
            best = dict(edge_idx=i, alpha=t_clamped, proj=proj_i, d2=d2_i,
                        tangent=t_hat, normal=n_hat, p0=p0_i, p1=p1_i)

    dist = float(np.sqrt(best["d2"])) if best is not None else np.inf
    is_on = bool(dist <= tol)
    v_e = best["p1"] - best["p0"]
    best.update({"is_on_building": is_on, "distance": dist, "v_e": v_e})
    return best


def build_defender_geometry(map_in, cam_dict):
    cam_x_direc = cam_dict['directional']['x']
    cam_y_direc = cam_dict['directional']['y']
    cam_x_omni = cam_dict['omnidirectional']['x']
    cam_y_omni = cam_dict['omnidirectional']['y']

    building_edge_vec, building_vector_vec = [], []

    for n_building in range(map_in['st']['n']):
        for n_sensor in range(cam_dict['n_direc']):
            pos = closest_edge(map_in['st'][str(n_building)],
                                [cam_x_direc[n_sensor], cam_y_direc[n_sensor]])
            if pos['is_on_building']:
                building_edge_vec.append(pos['p0'])
                building_vector_vec.append(pos['v_e'])

    for n_building in range(map_in['st']['n']):
        for n_sensor in range(cam_dict['n_omni']):
            pos = closest_edge(map_in['st'][str(n_building)],
                                [cam_x_omni[n_sensor], cam_y_omni[n_sensor]])
            if pos['is_on_building']:
                building_edge_vec.append(pos['p0'])
                building_vector_vec.append(pos['v_e'])

    return building_edge_vec, building_vector_vec


def build_camera_objects(cam_dict):
    camera_objects = []
    for i in range(cam_dict['n_direc']):
        camera_objects.append(Camera(i, cam_dict, cam_dict['directional']['x'][i],
                                      cam_dict['directional']['y'][i]))
    return camera_objects


def stage_detectability(x, y, t, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni):
    k_prod = 1.0
    for cam_i in range(cam_dict['n']):
        dx = x - S[0, cam_i]
        dy = y - S[1, cam_i]
        dist_sq = dx ** 2 + dy ** 2
        dist_min_sq = ca.fmax(dist_sq, 1e-4)
        dist_safe = ca.sqrt(dist_min_sq)

        if cam_i < cam_dict['n_direc']:
            cam = camera_objects[cam_i]
            phi = cam.get_ctr_theta_t(t)
            theta = cam.fov_ang
            cos_alpha = (dx * ca.cos(phi) + dy * ca.sin(phi)) / dist_safe
            visibility = 1 / (1 + ca.exp(-c_param * (cos_alpha - ca.cos(theta / 2))))
            observability = 1 / (1 + alpha_direc * dist_min_sq)
        else:
            visibility = 1
            observability = 1 / (1 + alpha_omni * dist_min_sq)

        k_j = visibility * observability
        k_prod *= (1 - k_j)

    return 1 - k_prod


def C_AS(A, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni, eps=1e-9):
    N = A.shape[1]
    C = 0
    for i in range(N):
        x_i, y_i, t_i = A[0, i], A[1, i], A[2, i]
        k_i = stage_detectability(x_i, y_i, t_i, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni)
        C += -ca.log(1.0 - k_i + eps)
    return C


def optimize_defender(A_fixed, cam_dict, building_edge_vec, building_vector_vec,
                       camera_objects, c_param, alpha_direc, alpha_omni):
    opti_d = ca.Opti()
    total_number_of_sensors = cam_dict['n']

    alpha_j = opti_d.variable(total_number_of_sensors)
    opti_d.subject_to(alpha_j >= 0)
    opti_d.subject_to(alpha_j <= 1)

    Sx, Sy = [], []
    for j in range(total_number_of_sensors):
        p1 = building_edge_vec[j]
        s = p1 + alpha_j[j] * building_vector_vec[j]
        Sx.append(s[0])
        Sy.append(s[1])
    S = ca.vertcat(ca.hcat(Sx), ca.hcat(Sy))

    N_eval_defender = A_fixed.shape[1]
    A_param = opti_d.parameter(3, N_eval_defender)
    opti_d.set_value(A_param, A_fixed)

    C = C_AS(A_param, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni)
    opti_d.minimize(-C)
    opti_d.set_initial(alpha_j, 0.5 * np.ones(total_number_of_sensors))

    p_opts = {"expand": False, "verbose": False, "print_time": False}
    s_opts = {'max_iter': 3000, 'tol': 1e-6, 'acceptable_tol': 1e-4, 'print_level': 0,
              "sb": "yes", "print_timing_statistics": "no"}
    opti_d.solver('ipopt', p_opts, s_opts)

    try:
        sol_d = opti_d.solve()
        alpha_star = sol_d.value(alpha_j)
        S_star = sol_d.value(S)
    except RuntimeError as e:
        print('optimize_defender failed:', e)
        alpha_star = opti_d.debug.value(alpha_j)
        S_star = opti_d.debug.value(S)
        sol_d = opti_d.debug
    return alpha_star, S_star, sol_d


def interpolate_rrt_path_preserve_speed(rrt_path, num_points):
    rrt_array = np.array(rrt_path)
    x, y, t = rrt_array[:, 0], rrt_array[:, 1], rrt_array[:, 2]
    fx = interp1d(t, x, kind='linear')
    fy = interp1d(t, y, kind='linear')
    t_interp = np.linspace(t[0], t[-1], num_points)
    return fx(t_interp).tolist(), fy(t_interp).tolist(), t_interp.tolist()


def optimize_attacker(S_star, path_ref, x0, xf, vmax, map_size, cam_dict,
                       obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni, N=250,
                       warm_start=None):
    """warm_start: optional (x_guess, y_guess, T_guess) to seed the NLP instead
    of path_ref's interpolation. x_guess/y_guess must have length N+1."""
    start_time = path_ref[0][2]
    end_time = path_ref[-1][2]

    opti_a = ca.Opti()
    T = opti_a.variable()
    dt = T / N
    opti_a.subject_to(T >= 1e-3)
    opti_a.subject_to(T <= end_time * 1.5)

    x_a = opti_a.variable(N + 1)
    y_a = opti_a.variable(N + 1)

    opti_a.subject_to(x_a[0] == x0[0])
    opti_a.subject_to(y_a[0] == x0[1])
    opti_a.subject_to(x_a[-1] == xf[0])
    opti_a.subject_to(y_a[-1] == xf[1])
    opti_a.subject_to(opti_a.bounded(map_size[0], x_a, map_size[1]))
    opti_a.subject_to(opti_a.bounded(map_size[2], y_a, map_size[3]))

    for Ni in range(N):
        dx = x_a[Ni + 1] - x_a[Ni]
        dy = y_a[Ni + 1] - y_a[Ni]
        opti_a.subject_to(dx ** 2 + dy ** 2 <= (vmax * dt) ** 2)

    S_param = opti_a.parameter(2, cam_dict['n'])
    opti_a.set_value(S_param, S_star)

    log_sum = 0
    gamma = 500
    safety_margin = 1 + 1  # vehicle_radius + buffer_dist
    for Ni in range(N):
        p_k = ca.vertcat(x_a[Ni], y_a[Ni])
        t_i = start_time + Ni * dt
        k_Ni = stage_detectability(x_a[Ni], y_a[Ni], t_i, S_param, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni)
        log_sum += ca.log(1 - k_Ni + 1e-9)

        for obs_data in obstacle_constraints:
            A_obs, b_obs = obs_data['A'], obs_data['b']
            h_i_vector = ca.mtimes(A_obs, p_k) - b_obs
            num_edges = A_obs.shape[0]
            max_h = h_i_vector[0]
            for idx in range(1, num_edges):
                max_h = ca.fmax(max_h, h_i_vector[idx])
            lse_term = max_h + (1.0 / gamma) * ca.log(ca.sum(ca.exp(gamma * (h_i_vector - max_h))))
            lse_debiased = lse_term - (ca.log(num_edges) / gamma)
            opti_a.subject_to(lse_debiased >= safety_margin)

    time_penalty = 0.02 * T
    J_total_cost = log_sum - time_penalty
    opti_a.minimize(-J_total_cost)

    if warm_start is not None:
        x_interp, y_interp, T_init = warm_start
    else:
        x_interp, y_interp, t_interp = interpolate_rrt_path_preserve_speed(path_ref, N + 1)
        T_init = float(path_ref[-1][2] - path_ref[0][2])
    opti_a.set_initial(x_a, x_interp)
    opti_a.set_initial(y_a, y_interp)
    opti_a.set_initial(T, T_init)

    p_opts = {"expand": True, "verbose": False, "print_time": False}
    s_opts = {'max_iter': 3000, 'tol': 1e-6, 'acceptable_tol': 1e-4, 'print_level': 0,
              "sb": "yes", "print_timing_statistics": "no"}
    opti_a.solver('ipopt', p_opts, s_opts)

    try:
        sol_a = opti_a.solve()
        x_a_opt, y_a_opt, T_opt = sol_a.value(x_a), sol_a.value(y_a), float(sol_a.value(T))
    except RuntimeError as e:
        print('optimize_attacker failed:', e)
        x_a_opt, y_a_opt, T_opt = opti_a.debug.value(x_a), opti_a.debug.value(y_a), float(opti_a.debug.value(T))

    t_opt_num = np.linspace(start_time, start_time + T_opt, N + 1)
    return np.vstack((x_a_opt.T, y_a_opt.T, t_opt_num.T))


def run_stp_rrt(map_in, cam_dict, dmap, x0, xf, vmax, map_size, vehicle, compTimeLimit=300, nPartition=5):
    while True:
        rrt_alg = rrt(vmax=vmax, xf=xf, map_size=map_size, vehicle=vehicle, cam_dict=cam_dict,
                      dmap=dmap, max_time=1, map_in=map_in)
        min_dist = np.sqrt((x0[0] - xf[0]) ** 2 + (x0[1] - xf[1]) ** 2)
        tf0 = min_dist / vmax
        tfn = 25
        _, _, _, _, _, path, _, _, _, _ = rrt_alg.standard_Parallel_RRT(
            x0, nPartition, compTimeLimit, tf0, tfn, debug=False)
        if path is not None:
            return path


def run_bilevel(path0, x0, xf, vmax, map_size, cam_dict, building_edge_vec, building_vector_vec,
                 obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni,
                 max_iters=50, hard_break_iter=20, N_attk=250, eta=None, warm_start_attacker=False,
                 cycle_window=4, tol_cost_windowed=1e-3, tol_sensor_windowed=0.1):
    """Faithful port of the alternating loop in cell 905e00e1 (with the
    S_prev -> S_star attacker-cost logging fix already applied).

    eta: if not None, applies Krasnoselskii-Mann damping to BOTH players'
    updates: x_{k+1} = (1-eta)*x_k + eta*BestResponse(x_k). eta=None
    reproduces the original undamped alternating best-response exactly.
    Damping S in Cartesian space is exactly equivalent to damping alpha_j
    in [0,1] edge-parameter space, since S = p0 + alpha*v is affine in
    alpha and (1-eta)+eta=1.

    warm_start_attacker: if True, seed optimize_attacker's NLP from the
    PREVIOUS iteration's (damped) attacker solution instead of always
    cold-restarting from path0's interpolation. Diagnostic 2 showed the
    cold-restart is extremely sensitive to tiny S changes (chaotic jumps
    to unrelated local optima) while warm-starting is smooth and stable.

    cycle_window / tol_*_windowed: a trailing-window convergence check
    (converged_windowed) alongside the original strict single-iteration
    check (converged). See docstring note above is_stable_windowed below.
    """
    path_arr = np.array(path0)
    A_prev = path_arr.T

    M = cam_dict['n']
    alpha_init_vec = 0.5 * np.ones(M)
    S_prev = np.column_stack([
        building_edge_vec[j] + alpha_init_vec[j] * building_vector_vec[j] for j in range(M)
    ])

    prev_total_cost = 0
    A_star, S_star = A_prev, S_prev
    converged_windowed = False
    converged = False
    n_iters_run = 0
    trace = []
    attacker_warm = None  # (x_guess, y_guess, T_guess), set after first dense A_star

    for it in range(max_iters):
        try:
            alpha_star, S_star_raw, sol_d = optimize_defender(
                A_prev, cam_dict, building_edge_vec, building_vector_vec,
                camera_objects, c_param, alpha_direc, alpha_omni)
        except RuntimeError as e:
            print('defender step failed:', e)
            continue
        S_star = S_star_raw if eta is None else (1 - eta) * S_prev + eta * S_star_raw

        try:
            A_star_raw = optimize_attacker(S_star, path0, x0, xf, vmax, map_size, cam_dict,
                                            obstacle_constraints, camera_objects, c_param,
                                            alpha_direc, alpha_omni, N=N_attk,
                                            warm_start=attacker_warm if warm_start_attacker else None)
        except RuntimeError as e:
            print('attacker step failed:', e)
            continue
        # A_prev is the sparse raw RRT path (few waypoints) only on the very
        # first iteration; from there on it matches A_star_raw's dense
        # (3, N_attk+1) shape. Skip damping when shapes don't match yet.
        if eta is None or A_prev.shape != A_star_raw.shape:
            A_star = A_star_raw
        else:
            A_star = (1 - eta) * A_prev + eta * A_star_raw
        attacker_warm = (A_star[0], A_star[1], float(A_star[2, -1] - A_star[2, 0]))

        n_iters_run = it + 1
        val_total = float(C_AS(A_star, S_star, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))

        current_total_cost = val_total
        cost_diff = abs(current_total_cost - prev_total_cost)
        sensor_shift = np.linalg.norm(S_star - S_prev)
        print(f'    iter {it}: cost={val_total:.4f}  cost_diff={cost_diff:.6f}  sensor_shift={sensor_shift:.6f}')
        trace.append({'iter': it, 'val_total': val_total, 'cost_diff': cost_diff, 'sensor_shift': sensor_shift})

        is_stable = (sensor_shift < 1e-3) and (cost_diff < 1e-4)

        # Windowed convergence check: require the last cycle_window iterations
        # to ALL satisfy relaxed tolerances, instead of one single iteration
        # hitting the strict threshold. Diagnostic (2026-08-11) showed
        # warm-started+damped trials often settle into a tiny residual
        # jitter (sensor_shift ~0.01-0.1, still >1e-3) around a near-flat
        # direction in sensor-placement space while cost is essentially
        # stable -- the single-shot AND-threshold is brittle to that jitter.
        is_stable_windowed = False
        if len(trace) >= cycle_window:
            recent = trace[-cycle_window:]
            max_cost_diff_w = max(t['cost_diff'] for t in recent)
            max_sensor_shift_w = max(t['sensor_shift'] for t in recent)
            is_stable_windowed = (max_cost_diff_w < tol_cost_windowed) and (max_sensor_shift_w < tol_sensor_windowed)

        if is_stable:
            converged = True
            converged_windowed = True
            break
        elif is_stable_windowed:
            converged_windowed = True
            break
        elif it >= hard_break_iter:
            break
        else:
            prev_total_cost = current_total_cost

        A_prev = A_star.copy()
        S_prev = S_star.copy()

    val_att_final = float(C_AS(A_star, S_star, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
    return A_star, S_star, converged, n_iters_run, val_att_final, trace, converged_windowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-trials', type=int, default=6)
    ap.add_argument('--k-alt', type=int, default=3)
    ap.add_argument('--n-attk', type=int, default=200)
    ap.add_argument('--max-iters', type=int, default=50)
    ap.add_argument('--hard-break-iter', type=int, default=20)
    ap.add_argument('--eta', type=float, default=None, help='damping factor in (0,1]; omit for undamped (original) behavior')
    ap.add_argument('--warm-start-attacker', action='store_true', help='seed optimize_attacker from the previous iterate instead of cold-restarting from path0')
    ap.add_argument('--cycle-window', type=int, default=4)
    ap.add_argument('--tol-cost-windowed', type=float, default=1e-3)
    ap.add_argument('--tol-sensor-windowed', type=float, default=0.1)
    ap.add_argument('--rng-seed', type=int, default=2)
    ap.add_argument('--out-dir', type=str, default='.')
    args = ap.parse_args()

    map_size_world = (100, 100)
    num_buildings = 10
    num_direc_sensor = 10
    num_omni_sensor = 10
    fov_half_rad = np.deg2rad(15)
    max_range = 20
    panspeed_range = (np.deg2rad(-10), np.deg2rad(10))
    cam_period = 200
    cam_increment = 0.2

    map_in, cam_dict = generate_map(
        map_size=map_size_world, num_buildings=num_buildings,
        num_directional_sensors=num_direc_sensor, num_omni_sensors=num_omni_sensor,
        fov_half_rad=fov_half_rad, max_range=max_range,
        fov_half_rad_omni=fov_half_rad, max_range_omni=max_range / 2,
        panspeed_range=panspeed_range, cam_period=cam_period, cam_increment=cam_increment,
        rng_seed=args.rng_seed, balanced_sensors=False,
        param_lambda=[0.5, 0.5], param_beta=[0.4, 0.25])
    map_size = map_in['st']['size']

    obstacle_constraints = build_obstacle_constraints(map_in)
    building_edge_vec, building_vector_vec = build_defender_geometry(map_in, cam_dict)
    camera_objects = build_camera_objects(cam_dict)

    alpha_direc = 99 / (cam_dict['directional']['spec']['fov'][1] ** 2)
    alpha_omni = 99 / (cam_dict['omnidirectional']['spec']['fov'][1] ** 2)
    c_param = 10

    x0 = [0, 0, 0]
    xf = [100, 100, 360]

    results = []
    t_start_all = time.time()

    for trial in range(args.num_trials):
        t0 = time.time()
        vmax = rn.uniform(10, 20)
        print(f'\n=== Trial {trial} | vmax={vmax:.3f} ===')

        dmap = DynamicMap(map_in, cam_dict)
        map_in['dy'] = dmap
        vehicle = {'v': vmax, 'radius': 2}  # match optimize_attacker's safety_margin (vehicle_radius=1 + buffer_dist=1)

        path0 = run_stp_rrt(map_in, cam_dict, dmap, x0, xf, vmax, map_size, vehicle)
        print(f'  initial RRT* path: {len(path0)} waypoints')

        A_star, S_star, converged, n_iters, cost_converged, trace, converged_windowed = run_bilevel(
            path0, x0, xf, vmax, map_size, cam_dict, building_edge_vec, building_vector_vec,
            obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni,
            max_iters=args.max_iters, hard_break_iter=args.hard_break_iter, N_attk=args.n_attk,
            eta=args.eta, warm_start_attacker=args.warm_start_attacker,
            cycle_window=args.cycle_window, tol_cost_windowed=args.tol_cost_windowed,
            tol_sensor_windowed=args.tol_sensor_windowed)
        print(f'  bilevel converged={converged} (windowed={converged_windowed}) in {n_iters} iters, cost*={cost_converged:.4f}')

        alt_costs = []
        alt_paths = []
        for kk in range(args.k_alt):
            path_alt = run_stp_rrt(map_in, cam_dict, dmap, x0, xf, vmax, map_size, vehicle)
            A_alt = optimize_attacker(S_star, path_alt, x0, xf, vmax, map_size, cam_dict,
                                       obstacle_constraints, camera_objects, c_param,
                                       alpha_direc, alpha_omni, N=args.n_attk)
            cost_alt = float(C_AS(A_alt, S_star, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
            alt_costs.append(cost_alt)
            alt_paths.append(A_alt)
            print(f'  alt seed {kk}: cost={cost_alt:.4f}')

        best_alt = min(alt_costs) if alt_costs else None
        beat = (best_alt is not None) and (best_alt < cost_converged - 1e-3)
        margin_pct = (100.0 * (cost_converged - best_alt) / cost_converged) if beat else 0.0

        # 1-B: multi-homotopy attacker initialization. gradient-based
        # refinement can only move within the homotopy class of its RRT*
        # seed (see diagnostic_2 notes); a K-seed search after convergence
        # lets the attacker escape a locally-good-but-globally-suboptimal
        # route. Validated 2026-08-11: beat rate 100% (3/3), mean margin
        # ~10% even against the now-properly-converging bilevel pipeline.
        if beat:
            best_kk = int(np.argmin(alt_costs))
            print(f'  1-B: multi-homotopy found a better route (alt seed {best_kk}, '
                  f'{cost_converged:.4f} -> {best_alt:.4f}, {margin_pct:.2f}% better) -- adopting it as the final answer')
            A_star = alt_paths[best_kk]
            cost_converged = best_alt

        elapsed = time.time() - t0
        print(f'  best_alt={best_alt}, beat={beat}, margin={margin_pct:.2f}%, trial_time={elapsed:.1f}s')

        results.append({
            'trial': trial,
            'vmax': vmax,
            'converged': converged,
            'converged_windowed': converged_windowed,
            'n_iters': n_iters,
            'cost_converged': cost_converged,
            'trace': trace,
            'alt_costs': alt_costs,
            'best_alt_cost': best_alt,
            'beat': beat,
            'margin_pct': margin_pct,
            'trial_time_sec': elapsed,
        })

        # incremental save so partial progress survives if we need to stop early
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out_dir, f'{timestamp}_1A_homotopy_diagnostic_partial.json')
        with open(out_path, 'w') as f:
            json.dump(make_json_serializable({'args': vars(args), 'results': results}), f, indent=2)

    n_beat = sum(1 for r in results if r['beat'])
    beat_rate = 100.0 * n_beat / len(results) if results else 0.0
    margins = [r['margin_pct'] for r in results if r['beat']]

    n_converged = sum(1 for r in results if r['converged'])
    n_converged_windowed = sum(1 for r in results if r['converged_windowed'])

    summary = {
        'num_trials': len(results),
        'converged_strict_rate_pct': 100.0 * n_converged / len(results) if results else 0.0,
        'converged_windowed_rate_pct': 100.0 * n_converged_windowed / len(results) if results else 0.0,
        'beat_rate_pct': beat_rate,
        'n_beat': n_beat,
        'mean_margin_pct_when_beat': float(np.mean(margins)) if margins else 0.0,
        'max_margin_pct_when_beat': float(np.max(margins)) if margins else 0.0,
        'total_time_sec': time.time() - t_start_all,
    }
    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f'{timestamp}_1A_homotopy_diagnostic.json')
    with open(out_path, 'w') as f:
        json.dump(make_json_serializable({'args': vars(args), 'summary': summary, 'results': results}), f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
