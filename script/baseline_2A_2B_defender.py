"""
2-A / 2-B baselines (RAL_Review_Fix_0810.md), addressing R3's "why is
continuous placement necessary" and R2/R3's "baseline is only random
placement" complaints.

For a fixed attacker trajectory, compares four defender placement
strategies, all evaluated with the SAME C_AS(A_fixed, S) detection-cost
function (higher = better for the defender):

  1. random        - uniform random alpha_j in [0,1] (the ONLY baseline
                      in the original submission)
  2. discretized    - 2-A: alpha_j restricted to K candidate points per
                      edge, chosen via coordinate ascent (discrete analogue
                      of the continuous NLP)
  3. coverage       - 2-B: sensors placed to maximize a static, attacker-
                      agnostic, time-averaged domain-coverage objective
                      (greedy/coordinate-ascent submodular-style heuristic),
                      then evaluated against the actual attacker path
  4. continuous_nlp - the existing method's single-shot defender best
                      response (optimize_defender), i.e. an upper bound on
                      what continuous placement can achieve against this
                      one fixed attacker path
"""

import argparse
import json
import os
import random as rn
import time
from datetime import datetime

import numpy as np
from shapely.geometry import Point, Polygon

from diagnostic_1A_homotopy_check import (
    generate_map, build_obstacle_constraints, build_defender_geometry,
    build_camera_objects, C_AS, optimize_defender, run_stp_rrt,
)
from Dynamic import DynamicMap
from json_seriablizable import make_json_serializable


# ----------------------------------------------------------------------
# Shared candidate-set / coordinate-ascent machinery (2-A and 2-B both
# use this, just with different objective functions)
# ----------------------------------------------------------------------

def build_candidates(building_edge_vec, building_vector_vec, K):
    """candidates[j] = list of K (alpha, xy) pairs evenly spaced along
    sensor j's assigned building edge."""
    alphas = np.linspace(0.0, 1.0, K)
    candidates = []
    for p0, v in zip(building_edge_vec, building_vector_vec):
        candidates.append([(a, p0 + a * v) for a in alphas])
    return candidates


def coordinate_ascent(objective_fn, candidates, M, num_sweeps=3, init_idx=None):
    """Generic discretized coordinate ascent: repeatedly, for each sensor,
    try all its candidate positions (holding the rest fixed) and keep the
    best. objective_fn(S) is MAXIMIZED. S has shape (2, M)."""
    if init_idx is None:
        init_idx = [len(candidates[j]) // 2 for j in range(M)]  # start at alpha=0.5, matches NLP init
    idx = list(init_idx)

    def build_S(idx):
        return np.column_stack([candidates[j][idx[j]][1] for j in range(M)])

    S = build_S(idx)
    best_val = objective_fn(S)

    for sweep in range(num_sweeps):
        improved = False
        for j in range(M):
            best_j_val, best_j_idx = best_val, idx[j]
            for k in range(len(candidates[j])):
                if k == idx[j]:
                    continue
                trial_idx = idx.copy()
                trial_idx[j] = k
                val = objective_fn(build_S(trial_idx))
                if val > best_j_val:
                    best_j_val, best_j_idx = val, k
            if best_j_idx != idx[j]:
                idx[j] = best_j_idx
                best_val = best_j_val
                improved = True
        if not improved:
            break

    alpha_star = np.array([candidates[j][idx[j]][0] for j in range(M)])
    return build_S(idx), alpha_star, best_val


# ----------------------------------------------------------------------
# 2-B: static, attacker-agnostic, time-averaged coverage objective
# ----------------------------------------------------------------------

def build_domain_grid(map_in, n_side=20):
    (x0, x1, y0, y1) = map_in['st']['size']
    xs = np.linspace(x0, x1, n_side)
    ys = np.linspace(y0, y1, n_side)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    polys = [Polygon([tuple(v) for v in map_in['st'][str(i)]]) for i in range(map_in['st']['n'])]
    keep = np.array([not any(poly.contains(Point(p)) for poly in polys) for p in pts])
    return pts[keep]


def camera_static_params(cam_dict, camera_objects):
    n_direc = cam_dict['n_direc']
    tilt_lim = np.array([camera_objects[i].tilt_lim for i in range(n_direc)], dtype=float)
    init_angle = np.array([camera_objects[i].init_angle for i in range(n_direc)], dtype=float)
    fov_ang = np.array([camera_objects[i].fov_ang for i in range(n_direc)], dtype=float)
    cam_fovspeed = np.array([camera_objects[i].cam_fovspeed for i in range(n_direc)], dtype=float)
    return tilt_lim, init_angle, fov_ang, cam_fovspeed


def coverage_objective_numpy(S, grid_xy, t_samples, cam_dict, cam_static, c_param, alpha_direc, alpha_omni):
    """Pure-numpy re-implementation of the stage_detectability/C_AS combine
    rule, evaluated over a static domain grid and averaged over t_samples
    (approximating a full directional-sensor pan sweep). Returns the mean
    per-grid-point detection probability -- higher = more of the domain is
    covered on average. No CasADi/NLP involved; this is a plain scoring
    function used only for the coordinate-ascent heuristic."""
    tilt_lim, init_angle, fov_ang, cam_fovspeed = cam_static
    n_direc = cam_dict['n_direc']

    total = np.zeros(grid_xy.shape[0])
    for t in t_samples:
        dx = grid_xy[:, 0][:, None] - S[0, :][None, :]
        dy = grid_xy[:, 1][:, None] - S[1, :][None, :]
        dist_sq = dx ** 2 + dy ** 2
        dist_min_sq = np.maximum(dist_sq, 1e-4)
        dist_safe = np.sqrt(dist_min_sq)

        mid = (tilt_lim[:, 0] + tilt_lim[:, 1]) / 2
        amp = (tilt_lim[:, 1] - tilt_lim[:, 0]) / 2
        phi = init_angle + mid + amp * np.sin(cam_fovspeed * t)
        cos_alpha = (dx[:, :n_direc] * np.cos(phi)[None, :] + dy[:, :n_direc] * np.sin(phi)[None, :]) / dist_safe[:, :n_direc]
        visibility_d = 1 / (1 + np.exp(-c_param * (cos_alpha - np.cos(fov_ang / 2)[None, :])))
        observability_d = 1 / (1 + alpha_direc * dist_min_sq[:, :n_direc])
        k_d = visibility_d * observability_d

        observability_o = 1 / (1 + alpha_omni * dist_min_sq[:, n_direc:])
        k_o = observability_o

        k_all = np.concatenate([k_d, k_o], axis=1)
        k_prod = np.prod(1 - k_all, axis=1)
        total += (1 - k_prod)

    return float(np.mean(total / len(t_samples)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-trials', type=int, default=10)
    ap.add_argument('--k-candidates', type=int, default=10)
    ap.add_argument('--num-sweeps', type=int, default=3)
    ap.add_argument('--grid-side', type=int, default=20)
    ap.add_argument('--t-samples', type=int, default=8)
    ap.add_argument('--rng-seed', type=int, default=2, help='seeds map/sensor generation -- keep identical across parallel workers to share one environment')
    ap.add_argument('--seed-offset', type=int, default=0, help='seeds this process\'s per-trial randomness -- use a distinct value per parallel worker')
    ap.add_argument('--out-dir', type=str, default='.')
    ap.add_argument('--out-prefix', type=str, default='', help='prepended to output filenames -- avoids collisions when running parallel workers into the same --out-dir')
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

    # Seed per-trial randomness AFTER map generation, so parallel workers
    # sharing --rng-seed still get the same environment but distinct,
    # reproducible, non-overlapping trial sequences.
    rn.seed(args.seed_offset)
    np.random.seed(args.seed_offset)

    obstacle_constraints = build_obstacle_constraints(map_in)
    building_edge_vec, building_vector_vec = build_defender_geometry(map_in, cam_dict)
    camera_objects = build_camera_objects(cam_dict)
    alpha_direc = 99 / (cam_dict['directional']['spec']['fov'][1] ** 2)
    alpha_omni = 99 / (cam_dict['omnidirectional']['spec']['fov'][1] ** 2)
    c_param = 10
    M = cam_dict['n']

    candidates = build_candidates(building_edge_vec, building_vector_vec, args.k_candidates)

    print('Building domain grid for 2-B coverage objective...')
    grid_xy = build_domain_grid(map_in, n_side=args.grid_side)
    print(f'  grid: {grid_xy.shape[0]} points (outside buildings)')
    cam_static = camera_static_params(cam_dict, camera_objects)
    # sample times across a representative sweep window
    t_samples = np.linspace(0, 60, args.t_samples)

    def obj_coverage(S):
        return coverage_objective_numpy(S, grid_xy, t_samples, cam_dict, cam_static, c_param, alpha_direc, alpha_omni)

    print('Running 2-B coordinate ascent (attacker-agnostic coverage)...')
    t0 = time.time()
    S_coverage, alpha_coverage, coverage_score = coordinate_ascent(obj_coverage, candidates, M, num_sweeps=args.num_sweeps)
    print(f'  done in {time.time()-t0:.1f}s, coverage_score={coverage_score:.4f}')

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

        # 1. random placement (matches the ONLY baseline in the original submission)
        alpha_random = np.random.uniform(0, 1, M)
        S_random = np.column_stack([building_edge_vec[j] + alpha_random[j] * building_vector_vec[j] for j in range(M)])
        cost_random = float(C_AS(A_fixed, S_random, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))

        # 2. discretized (2-A)
        def obj_2A(S):
            return float(C_AS(A_fixed, S, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        S_2A, alpha_2A, cost_2A = coordinate_ascent(obj_2A, candidates, M, num_sweeps=args.num_sweeps)

        # 3. coverage heuristic (2-B), computed once above (attacker-agnostic),
        #    evaluated here against this trial's actual attacker path
        cost_coverage = float(C_AS(A_fixed, S_coverage, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))

        # 4. continuous NLP (existing method's single-shot defender best response)
        try:
            _, S_nlp, sol_d = optimize_defender(A_fixed, cam_dict, building_edge_vec, building_vector_vec,
                                                 camera_objects, c_param, alpha_direc, alpha_omni)
            cost_nlp = float(C_AS(A_fixed, S_nlp, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        except RuntimeError as e:
            print('  continuous NLP failed:', e)
            cost_nlp = None

        elapsed = time.time() - t0
        print(f'  random={cost_random:.4f}  discretized(2A)={cost_2A:.4f}  coverage(2B)={cost_coverage:.4f}  continuous_nlp={cost_nlp}  ({elapsed:.1f}s)')

        results.append({
            'trial': trial, 'vmax': vmax,
            'cost_random': cost_random,
            'cost_discretized_2A': cost_2A,
            'cost_coverage_2B': cost_coverage,
            'cost_continuous_nlp': cost_nlp,
            'trial_time_sec': elapsed,
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out_dir, f'{args.out_prefix}{timestamp}_2A2B_baseline_partial.json')
        with open(out_path, 'w') as f:
            json.dump(make_json_serializable({'args': vars(args), 'coverage_score_static': coverage_score, 'results': results}), f, indent=2)

    def mean_of(key):
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        'num_trials': len(results),
        'mean_cost_random': mean_of('cost_random'),
        'mean_cost_discretized_2A': mean_of('cost_discretized_2A'),
        'mean_cost_coverage_2B': mean_of('cost_coverage_2B'),
        'mean_cost_continuous_nlp': mean_of('cost_continuous_nlp'),
    }
    print('\n=== SUMMARY (mean detection cost, higher = better for defender) ===')
    print(json.dumps(summary, indent=2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f'{args.out_prefix}{timestamp}_2A2B_baseline.json')
    with open(out_path, 'w') as f:
        json.dump(make_json_serializable({'args': vars(args), 'coverage_score_static': coverage_score, 'summary': summary, 'results': results}), f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
