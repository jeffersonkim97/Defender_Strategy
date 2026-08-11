"""
Diagnostic 2: Is the small-amplitude jitter (trials 0,1,6 with eta=0.3) real
game-dynamics oscillation, or an artifact of optimize_attacker always cold-
restarting from the ORIGINAL path0 interpolation instead of warm-starting
from the previous iterate?

Tests:
  A. Determinism: call optimize_attacker twice with IDENTICAL (S, path0) ->
     should be bit-identical if IPOPT is truly deterministic here.
  B. Sensitivity: perturb S by a tiny amount (comparable to what damping
     produces between iterations) -> does the attacker's solution change
     smoothly, or jump to a qualitatively different trajectory/cost?
  C. Warm-start: same tiny S perturbation, but this time seed the NLP's
     initial guess from the UNPERTURBED run's own solution instead of the
     generic path0 interpolation -> does that suppress the jump?
"""

import numpy as np
import random as rn

from diagnostic_1A_homotopy_check import (
    generate_map, build_obstacle_constraints, build_defender_geometry,
    build_camera_objects, C_AS, optimize_defender, optimize_attacker,
    run_stp_rrt, interpolate_rrt_path_preserve_speed,
)
from Dynamic import DynamicMap
import casadi as ca


def optimize_attacker_warmstart(S_star, path_ref, x0, xf, vmax, map_size, cam_dict,
                                 obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni,
                                 N, warm_x, warm_y, warm_T):
    """Same NLP as optimize_attacker but seeded from a caller-supplied
    (warm_x, warm_y, warm_T) initial guess instead of path_ref's interpolation."""
    import diagnostic_1A_homotopy_check as d1a
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
    safety_margin = 2
    for Ni in range(N):
        p_k = ca.vertcat(x_a[Ni], y_a[Ni])
        t_i = start_time + Ni * dt
        k_Ni = d1a.stage_detectability(x_a[Ni], y_a[Ni], t_i, S_param, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni)
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

    opti_a.set_initial(x_a, warm_x)
    opti_a.set_initial(y_a, warm_y)
    opti_a.set_initial(T, warm_T)

    p_opts = {"expand": True, "verbose": False, "print_time": False}
    s_opts = {'max_iter': 3000, 'tol': 1e-6, 'acceptable_tol': 1e-4, 'print_level': 0,
              "sb": "yes", "print_timing_statistics": "no"}
    opti_a.solver('ipopt', p_opts, s_opts)

    try:
        sol_a = opti_a.solve()
        x_a_opt, y_a_opt, T_opt = sol_a.value(x_a), sol_a.value(y_a), float(sol_a.value(T))
    except RuntimeError as e:
        print('warmstart solve failed:', e)
        x_a_opt, y_a_opt, T_opt = opti_a.debug.value(x_a), opti_a.debug.value(y_a), float(opti_a.debug.value(T))

    t_opt_num = np.linspace(start_time, start_time + T_opt, N + 1)
    return np.vstack((x_a_opt.T, y_a_opt.T, t_opt_num.T))


def main():
    map_size_world = (100, 100)
    map_in, cam_dict = generate_map(
        map_size=map_size_world, num_buildings=10,
        num_directional_sensors=10, num_omni_sensors=10,
        fov_half_rad=np.deg2rad(15), max_range=20,
        fov_half_rad_omni=np.deg2rad(15), max_range_omni=10,
        panspeed_range=(np.deg2rad(-10), np.deg2rad(10)), cam_period=200, cam_increment=0.2,
        rng_seed=2, balanced_sensors=False, param_lambda=[0.5, 0.5], param_beta=[0.4, 0.25])
    map_size = map_in['st']['size']

    obstacle_constraints = build_obstacle_constraints(map_in)
    building_edge_vec, building_vector_vec = build_defender_geometry(map_in, cam_dict)
    camera_objects = build_camera_objects(cam_dict)
    alpha_direc = 99 / (cam_dict['directional']['spec']['fov'][1] ** 2)
    alpha_omni = 99 / (cam_dict['omnidirectional']['spec']['fov'][1] ** 2)
    c_param = 10
    N = 200

    x0 = [0, 0, 0]
    xf = [100, 100, 360]
    vmax = 15.0  # fixed, mid-range

    dmap = DynamicMap(map_in, cam_dict)
    map_in['dy'] = dmap
    vehicle = {'v': vmax, 'radius': 2}

    print('Generating a reference attacker path (RRT*)...')
    path0 = run_stp_rrt(map_in, cam_dict, dmap, x0, xf, vmax, map_size, vehicle)
    print(f'  path0: {len(path0)} waypoints')

    path_arr = np.array(path0)
    A_ref = path_arr.T

    print('Solving reference defender best-response against path0...')
    alpha_star, S_ref, sol_d = optimize_defender(
        A_ref, cam_dict, building_edge_vec, building_vector_vec,
        camera_objects, c_param, alpha_direc, alpha_omni)
    print(f'  S_ref obtained, alpha_star={np.round(alpha_star, 3)}')

    # --- Test A: determinism ---
    print('\n=== TEST A: determinism (identical S, identical cold-start guess) ===')
    A1 = optimize_attacker(S_ref, path0, x0, xf, vmax, map_size, cam_dict,
                            obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni, N=N)
    A2 = optimize_attacker(S_ref, path0, x0, xf, vmax, map_size, cam_dict,
                            obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni, N=N)
    cost1 = float(C_AS(A1, S_ref, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
    cost2 = float(C_AS(A2, S_ref, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
    max_diff = float(np.max(np.abs(A1 - A2)))
    print(f'  cost1={cost1:.6f}  cost2={cost2:.6f}  max|A1-A2|={max_diff:.2e}')
    print('  -> DETERMINISTIC' if max_diff < 1e-9 else '  -> NOT bit-identical (some nondeterminism present)')

    # --- Test B: sensitivity to a tiny S perturbation (cold-start each time) ---
    print('\n=== TEST B: sensitivity to tiny S perturbation (cold-start guess each time) ===')
    rn_state = np.random.RandomState(42)
    for eps in [0.01, 0.05, 0.1, 0.3, 1.0]:
        S_pert = S_ref + rn_state.normal(scale=eps, size=S_ref.shape)
        A_pert = optimize_attacker(S_pert, path0, x0, xf, vmax, map_size, cam_dict,
                                    obstacle_constraints, camera_objects, c_param, alpha_direc, alpha_omni, N=N)
        cost_pert = float(C_AS(A_pert, S_pert, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        cost_pert_vs_Sref = float(C_AS(A_pert, S_ref, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        traj_diff = float(np.max(np.abs(A_pert[:2] - A1[:2])))
        print(f'  |S perturbation|~N(0,{eps}): cost={cost_pert:.4f} (baseline={cost1:.4f}, delta={cost_pert-cost1:+.4f})'
              f'  max trajectory |dx,dy| shift={traj_diff:.2f}')

    # --- Test C: same perturbations, but warm-start from the UNPERTURBED solution ---
    print('\n=== TEST C: same perturbations, warm-started from unperturbed A1 solution ===')
    T1 = float(A1[2, -1] - A1[2, 0])
    for eps in [0.01, 0.05, 0.1, 0.3, 1.0]:
        S_pert = S_ref + rn_state.normal(scale=eps, size=S_ref.shape)
        A_pert_ws = optimize_attacker_warmstart(
            S_pert, path0, x0, xf, vmax, map_size, cam_dict, obstacle_constraints, camera_objects,
            c_param, alpha_direc, alpha_omni, N, A1[0], A1[1], T1)
        cost_pert_ws = float(C_AS(A_pert_ws, S_pert, cam_dict, camera_objects, c_param, alpha_direc, alpha_omni))
        traj_diff_ws = float(np.max(np.abs(A_pert_ws[:2] - A1[:2])))
        print(f'  |S perturbation|~N(0,{eps}): cost={cost_pert_ws:.4f} (baseline={cost1:.4f}, delta={cost_pert_ws-cost1:+.4f})'
              f'  max trajectory |dx,dy| shift={traj_diff_ws:.2f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
