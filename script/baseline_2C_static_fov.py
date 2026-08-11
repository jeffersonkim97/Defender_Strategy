"""
2-C baseline (RAL_Review_Fix_0810.md): a defender that ignores directional
sensors' panning and treats each one as pointed at a FIXED angle (the
midpoint of its sweep range) instead of the true time-varying
phi(t) = init_angle + mid + amp*sin(w*t).

R2 explicitly asked for "a sensor placement algorithm which does not
consider moving camera FOV" as a stronger baseline than random placement.

For each trial we report three numbers (all via the SAME C_AS-style
detection cost, higher = better for the defender):
  cost_dynamic_real   - existing method: optimize placement against the
                         TRUE dynamic (panning) model, evaluated on the
                         true model. The "upper bound" / current method.
  cost_static_self    - naive method: optimize placement against a STATIC
                         (frozen-angle) model, evaluated on that SAME
                         static model. What the naive defender *believes*
                         it achieves.
  cost_static_real    - the naive method's placement evaluated against the
                         TRUE dynamic model. What actually happens.

cost_static_self - cost_static_real is the naive defender's "self-deception
gap": it overestimates its own performance because it never models the
camera's blind spots while panning away.
"""

import argparse
import json
import os
import random as rn
import time
from datetime import datetime

import numpy as np
import casadi as ca

from diagnostic_1A_homotopy_check import (
    generate_map, build_obstacle_constraints, build_defender_geometry,
    build_camera_objects, C_AS, optimize_defender, run_stp_rrt,
)
from Dynamic import DynamicMap
from json_seriablizable import make_json_serializable


def stage_detectability_static(x, y, t, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni):
    """Same combine rule as stage_detectability, but each directional
    sensor's heading is frozen at its sweep midpoint (init_angle + mid)
    instead of the true time-varying get_ctr_theta_t(t). t is unused;
    kept in the signature so this is a drop-in swap for stage_detectability."""
    k_prod = 1.0
    for cam_i in range(cam_dict['n']):
        dx = x - S[0, cam_i]
        dy = y - S[1, cam_i]
        dist_sq = dx ** 2 + dy ** 2
        dist_min_sq = ca.fmax(dist_sq, 1e-4)
        dist_safe = ca.sqrt(dist_min_sq)

        if cam_i < cam_dict['n_direc']:
            cam = camera_objects[cam_i]
            b1, b2 = cam.tilt_lim[0], cam.tilt_lim[1]
            phi_static = cam.init_angle + (b1 + b2) / 2  # frozen at sweep midpoint, no sin(w*t)
            theta = cam.fov_ang
            cos_alpha = (dx * ca.cos(phi_static) + dy * ca.sin(phi_static)) / dist_safe
            visibility = 1 / (1 + ca.exp(-c_param * (cos_alpha - ca.cos(theta / 2))))
            observability = 1 / (1 + alpha_direc * dist_min_sq)
        else:
            visibility = 1
            observability = 1 / (1 + alpha_omni * dist_min_sq)

        k_j = visibility * observability
        k_prod *= (1 - k_j)
    return 1 - k_prod


def C_AS_static(A, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni, eps=1e-9):
    N = A.shape[1]
    C = 0
    for i in range(N):
        x_i, y_i, t_i = A[0, i], A[1, i], A[2, i]
        k_i = stage_detectability_static(x_i, y_i, t_i, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni)
        C += -ca.log(1.0 - k_i + eps)
    return C


def optimize_defender_static(A_fixed, cam_dict, building_edge_vec, building_vector_vec,
                              camera_objects, c_param, alpha_direc, alpha_omni):
    """Same NLP structure as optimize_defender, but optimizes against the
    frozen-angle static model instead of the true panning model."""
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

    C = C_AS_static(A_param, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni)
    opti_d.minimize(-C)
    opti_d.set_initial(alpha_j, 0.5 * np.ones(total_number_of_sensors))

    p_opts = {"expand": False, "verbose": False, "print_time": False}
    s_opts = {'max_iter': 3000, 'tol': 1e-6, 'acceptable_tol': 1e-4, 'print_level': 0,
              "sb": "yes", "print_timing_statistics": "no"}
    opti_d.solver('ipopt', p_opts, s_opts)

    try:
        sol_d = opti_d.solve()
        S_star = sol_d.value(S)
    except RuntimeError as e:
        print('optimize_defender_static failed:', e)
        S_star = opti_d.debug.value(S)
    return S_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-trials', type=int, default=10)
    ap.add_argument('--rng-seed', type=int, default=2)
    ap.add_argument('--out-dir', type=str, default='.')
    args = ap.parse_args()

    map_size_world = (100, 100)
    map_in, cam_dict = generate_map(
        map_size=map_size_world, num_buildings=10,
        num_directional_sensors=10, num_omni_sensors=10,
        fov_half_rad=np.deg2rad(15), max_range=20,
        fov_half_rad_omni=np.deg2rad(15), max_range_omni=10,
        panspeed_range=(np.deg2rad(-10), np.deg2rad(10)), cam_period=200, cam_increment=0.2,
        rng_seed=args.rng_seed, balanced_sensors=False, param_lambda=[0.5, 0.5], param_beta=[0.4, 0.25])
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
    for trial in range(args.num_trials):
        t0 = time.time()
        vmax = rn.uniform(10, 20)
        print(f'\n=== Trial {trial} | vmax={vmax:.3f} ===')

        dmap = DynamicMap(map_in, cam_dict)
        map_in['dy'] = dmap
        vehicle = {'v': vmax, 'radius': 2}

        path = run_stp_rrt(map_in, cam_dict, dmap, x0, xf, vmax, map_size, vehicle)
        A_fixed = np.array(path).T
        print(f'  RRT path: {A_fixed.shape[1]} waypoints')

        # dynamic (current method): optimize against the true panning model
        try:
            _, S_dynamic, _ = optimize_defender(A_fixed, cam_dict, building_edge_vec, building_vector_vec,
                                                 camera_objects, c_param, alpha_direc, alpha_omni)
            cost_dynamic_real = float(C_AS(A_fixed, S_dynamic, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        except RuntimeError as e:
            print('  dynamic NLP failed:', e)
            cost_dynamic_real = None

        # static (naive, R2-requested baseline): optimize against frozen-angle model
        try:
            S_static = optimize_defender_static(A_fixed, cam_dict, building_edge_vec, building_vector_vec,
                                                 camera_objects, c_param, alpha_direc, alpha_omni)
            cost_static_self = float(C_AS_static(A_fixed, S_static, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
            cost_static_real = float(C_AS(A_fixed, S_static, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        except RuntimeError as e:
            print('  static NLP failed:', e)
            cost_static_self = cost_static_real = None

        elapsed = time.time() - t0
        print(f'  dynamic_real={cost_dynamic_real}  static_self={cost_static_self}  static_real={cost_static_real}  ({elapsed:.1f}s)')

        results.append({
            'trial': trial, 'vmax': vmax,
            'cost_dynamic_real': cost_dynamic_real,
            'cost_static_self': cost_static_self,
            'cost_static_real': cost_static_real,
            'trial_time_sec': elapsed,
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out_dir, f'{timestamp}_2C_static_fov_partial.json')
        with open(out_path, 'w') as f:
            json.dump(make_json_serializable({'args': vars(args), 'results': results}), f, indent=2)

    def mean_of(key):
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        'num_trials': len(results),
        'mean_cost_dynamic_real': mean_of('cost_dynamic_real'),
        'mean_cost_static_self': mean_of('cost_static_self'),
        'mean_cost_static_real': mean_of('cost_static_real'),
    }
    print('\n=== SUMMARY (mean detection cost, higher = better for defender) ===')
    print(json.dumps(summary, indent=2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f'{timestamp}_2C_static_fov.json')
    with open(out_path, 'w') as f:
        json.dump(make_json_serializable({'args': vars(args), 'summary': summary, 'results': results}), f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
