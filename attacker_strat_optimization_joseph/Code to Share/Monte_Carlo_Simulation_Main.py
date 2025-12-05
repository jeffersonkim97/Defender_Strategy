import matplotlib.pyplot as plt
import matplotlib.animation as animation
import STP_RRTStar
import Dynamic
import numpy as np
import casadi as ca
import matplotlib.patches as patches
from IPython.display import HTML
from scipy.interpolate import splprep, splev
import importlib
import csv
import json
from scipy.interpolate import interp1d
from Camera import Camera
import time
import pprint
import pandas as pd
import itertools

importlib.reload(Dynamic)
from Dynamic import DynamicMap
importlib.reload(STP_RRTStar)
from STP_RRTStar import STP_RRTStar as Rrt


def do_casadi_opt_sim(scenario_key, initial_path_string):
    # Importing simulation parameters and generated paths from Jefferson
 
    # 31 has 9 cameras and has an initial guess from "ParallelRRT"
    # 2 has 4 cameras and has an initial guess from "ParallelRRT" 

    # Defender and map info
    with open("2D_Comparison_Defender_Map.json", "r") as f:
        defender_data = json.load(f)

    # path info
    with open("2D_Comparison_Path.json", "r") as f:
        path_data = json.load(f)

    '''STP-RRT stuff'''
    # Attacker Position
    map_size_original = [0, 100, 0, 100] # [-x, x, -y, y]
    map_translate = [0, 0]
    map_size = [map_size_original[0]-map_translate[0], map_size_original[1]-map_translate[0], map_size_original[2]-map_translate[1], map_size_original[3]-map_translate[1]]
    x0_1 = [5, 5, 0] # Start [x, y, t]
    x0list = [x0_1] 
    xf = [95,95] # Goal [x, y] (t doesn't matter)

    # list of camera x and y positions
    # cam_x = [3, 6, 7, 5]
    # cam_y = [8, 4, 2, 1]
    cam_x = defender_data[scenario_key]["x"]
    cam_y = defender_data[scenario_key]["y"]


    dynamic_obstacle_pos = []

    for i in range(len(cam_x)):
        dynamic_obstacle_pos.append([cam_x[i], cam_y[i]])

    prox = [0.25, 0.1] # proximity for connection [position, time] 

    # General RRT Settings
    iter_max = 1000     # maximum number of iterations for the tree
    vmax = 2            # max velocity for each point ?
    max_time = 1        # max time difference for each point

    # Test case
    # Vehicle Spec
    vehicle = {}
    vehicle['v'] = vmax
    vehicle['radius'] = 1
    t = [10, 1, 500] # [camera_period, camera_increment, some unused variable]

    # Map
    map_in = {}
    # Static Map
    map_in['st'] = {}
    map_in['st']['size'] = np.array([map_size[0], map_size[1], map_size[2], map_size[3]])
    # Single building example
    # This is an imaginary cylinder around the camera tripod position
    # Collision radius r_cyl = 250 [mm]
    # [center_x, center_y, radius]
    r_cyl = 1
    map_in['st']['n'] = len(cam_x)
    for i in range(len(cam_x)):
        map_in['st'][str(i)] = np.array([cam_x[i], cam_y[i], r_cyl])

    # Dynamic Map
    # This is a continuous function that generates camera FOV coverages
    # Input is map_in, and time input t_in
    map_in['n'] = t[0] # camera period

    # Camera Position
    cam_dict = {}
    cam_dict['n'] = len(cam_x)
    map_in['ncam'] = len(cam_x)
    cam_dict['x'] = cam_x
    cam_dict['y'] = cam_y

    # For Monte Carlo simulation, this is all loaded from Jefferson's file:
    cam_dict['spec'] = {}
    cam_dict['spec']['init_angle'] = defender_data[scenario_key]["spec"]["init_angle"]
    cam_dict['spec']['bound'] = defender_data[scenario_key]["spec"]["bound"]              
    cam_dict['spec']['fov'] = defender_data[scenario_key]["spec"]["fov"]
    cam_dict['spec']['cam_time'] = defender_data[scenario_key]["spec"]["cam_time"]
    cam_dict['spec']['panspeed'] = defender_data[scenario_key]["spec"]["panspeed"]

    # Importing fov angle and range from Jefferson's results
    fov_ang = cam_dict['spec']['fov'][0]
    fov_rng = cam_dict['spec']['fov'][1]


    cameras_dict_copilot = []
    for i in range(cam_dict['n']):
        cam_x_dict = cam_dict['x'][i]
        cam_y_dict = cam_dict['y'][i]
        lower = cam_dict['spec']['bound'][i][0]
        upper = cam_dict['spec']['bound'][i][1]
        amplitude = 0.5 * (upper - lower)
        center = cam_dict['spec']['init_angle'][i] # ASSUMING THAT INITIAL ANGLE IS CENTER ANGLE
        period = cam_dict['spec']['cam_time'][0]
        frequency = 1 / period  # Use Hz for animation
        init_angle = cam_dict['spec']['init_angle'][i]
        # phase_offset = np.arctan2(init_angle - center, amplitude)
        phase_offset = 0 # Assuming that all cameras start at center of angular range
        panspeed = cam_dict['spec']['panspeed'][i]

        cameras_dict_copilot.append({
            'pos': [cam_x_dict, cam_y_dict],
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase_offset,
            'center': center,
            'range': fov_rng,
            'panspeed': panspeed
        })


    # Test dynamic map
    dmap = DynamicMap(map_in, cam_dict)
    map_in['dy'] = dmap

    # STP-RRT* Execution parameters
    x0 = x0list[0]          # start point

    min_dist = np.sqrt((x0_1[0]-xf[0])**2 + (x0_1[1]-xf[1])**2)
    min_time = min_dist/vmax

    end_point = [xf[0], xf[1], min_time*5] # min_time*5 so that casadi can still have some flexibility
   

    if initial_path_string == "naive":
        # Case 1: Naive path (straight line)
        path = [x0_1, end_point] 
    elif initial_path_string in path_data:
        # Case 2: RRT-type path (look up in JSON)
        # Use .get() to safely retrieve the path. 
        # If '2' is not found, 'path' will be set to None.
        path = path_data[initial_path_string].get(str(scenario_key))
    else:
        # Case 3: initial_path_string is not 'naive' and not a valid key in path_data
        path = None

    if path != None:
        # =============================================== CasADi Section =================================================

        # Cost function: at each time step, take the pointwise probability of detection k(x,t) where x is the
        # position and t is the time. 
        # P_detection = 1 - P_not_detected
        # P_not_detected = (1-k(x1,t1))*(1-k(x2,t2))*(1-k(x3,t3))*...
        # P_not_detected = PI(1-k(x_i, t_i))
        # log(P_not_detected) = log(PI(1-k(x_i, t_i)))
        # log(P_not_detected) = log(1-k(x_1, t_1) + log(1-k(x_2, t_2) + log(1-k(x_3, t_3) + ...
        # So, at each time step, take the pointwise detectability, then take the log of it, then add it to the running cost
        # CasADi should try to MAXIMIZE this summation of logarithms to minimize the probability of detection

        # Once you have the sum of the logarithms, here's how you convert back to a probability of detection:
        # log(1-k(x_1, t_1) + log(1-k(x_2, t_2) + log(1-k(x_3, t_3) + ... = log(P_not_detected)
        # exp(log(P_not_detected)) = P_not_detected
        # P_detected = 1 - P_not_detected


        # Gemini alpha calculation:
        # alpha ~= 99/range_max^2

        tic = time.time()

        alpha = 99/(fov_rng**2)# This is the distance decay variable (higher = more decay).
        # ^^ This is calculated so that anything outside of the fov_rng is 1% detectable or less
        k = 100 # This is the steepness of visibility transition (higher = sharper transition)


        # New version, uses call to Camera.py to find FOV center:
        def calculate_k_value(xi, yi, ti, camera_objects):
            total_cost = 0
            for cam in camera_objects:
                cam_x, cam_y = cam.cam_position()
                phi = cam.get_ctr_theta_t(ti)  # Use Camera.py method

                theta = cam.fov_ang
                dx = xi - cam_x
                dy = yi - cam_y
                dist2 = dx**2 + dy**2
                dist = ca.sqrt(dist2 + 1e-6)

                cos_alpha = (dx * ca.cos(phi) + dy * ca.sin(phi)) / dist
                visibility = 1 / (1 + ca.exp(-k * (cos_alpha - ca.cos(theta / 2))))
                observability = 1 / (1 + alpha * dist2)
                total_cost += visibility * observability
            return total_cost


        start_time = path[0][2]
        end_time = path[len(path)-1][2]

        opti = ca.Opti()

        # Set timestamps
        N = 1000  # number of time steps (tune this)

        # NEW: total travel time T is a decision variable (>= small positive)
        T = opti.variable()

        # Derived: per-step time (CasADi expression)
        dt = T / N

        opti.subject_to(T >= 1e-3)  # strictly positive
        opti.subject_to(T <= end_time)

        # Decision variables
        x = opti.variable(N+1)  # UAV x coords
        y = opti.variable(N+1)  # UAV y coords

        # Initial position constraints
        opti.subject_to(x[0] == x0_1[0])
        opti.subject_to(y[0] == x0_1[1])
        # Final position constraints
        opti.subject_to(x[-1] == xf[0])
        opti.subject_to(y[-1] == xf[1])

        # Map bounds
        opti.subject_to(opti.bounded(map_size[0], x, map_size[1]))
        opti.subject_to(opti.bounded(map_size[2], y, map_size[3]))

        # Max speed constraint
        vmax = vehicle['v']
        for i in range(N):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            opti.subject_to(dx*dx + dy*dy <= (vmax*dt)**2)

        # Generating Camera objects
        camera_objects = []
        for i in range(cam_dict['n']):
            cam_obj = Camera(i, cam_dict, cam_dict['x'][i], cam_dict['y'][i])
            camera_objects.append(cam_obj)

        # New cost (I guess maybe it's a "utility"):
        log_sum = 0
        for i in range(N):
            t_i = start_time + i * dt
            k_value = calculate_k_value(x[i], y[i], t_i, camera_objects) # generating pointwise detectability
            current_log_term = ca.log(1 - k_value)
            log_sum += current_log_term

        # Cost function: at each time step, take the pointwise probability of detection k(x,t) where x is the
        # position and t is the time. 
        # P_detection = 1 - P_not_detected
        # P_not_detected = (1-k(x1,t1))*(1-k(x2,t2))*(1-k(x3,t3))*...
        # P_not_detected = PI(1-k(x_i, t_i))
        # log(P_not_detected) = log(PI(1-k(x_i, t_i)))
        # log(P_not_detected) = log(1-k(x_1, t_1) + log(1-k(x_2, t_2) + log(1-k(x_3, t_3) + ...
        # So, at each time step, take the pointwise detectability, then take the log of it, then add it to the running cost
        # CasADi should try to MAXIMIZE this summation of logarithms to minimize the probability of detection
        # So, minimize the negative of the log_sum


        # time penalty to encourage shorter time
        time_penalty = 0.02*T
        # -0.013 is the minimum to make the path not look choppy and terrible in the open scenarios
        # -0.014 gets the dense scenario to go to the end when in open areas
        # -0.015 still has weird choppyness in the dense scenario
        # -0.016 "
        # -0.02 gets the dense scenario to go straight when in open space, it also slows down the casadi calculation,
        # I'm going with that for now ^^

        J_total_cost = log_sum - time_penalty # We want to maximize the log sum, so the time penalty should be signed opposite from log_sum

        opti.minimize(-J_total_cost) # We want to MAXIMIZE the log sum, so we minimize the negative of the total cost

        def interpolate_rrt_path_preserve_speed(rrt_path, num_points):
            """
            Interpolates the original STP-RRT path using time-based interpolation,
            preserving the original speed profile.

            Parameters:
            - rrt_path: list of [x, y, t] points from STP-RRT
            - num_points: number of interpolated points to generate

            Returns:
            - x_interp: list of interpolated x positions
            - y_interp: list of interpolated y positions
            - t_interp: list of interpolated time values
            """
            rrt_array = np.array(rrt_path)
            x = rrt_array[:, 0]
            y = rrt_array[:, 1]
            t = rrt_array[:, 2]

            # Create interpolation functions for x(t) and y(t)
            fx = interp1d(t, x, kind='linear')
            fy = interp1d(t, y, kind='linear')

            # Generate evenly spaced time values
            t_interp = np.linspace(t[0], t[-1], num_points)

            # Interpolate positions
            x_interp = fx(t_interp)
            y_interp = fy(t_interp)

            return x_interp.tolist(), y_interp.tolist(), t_interp.tolist()



        # Interpolating RRT initial guess
        x_interp, y_interp, t_interp = interpolate_rrt_path_preserve_speed(path, N+1)


        # Using time interpolated RRT as the initial guess
        opti.set_initial(x, x_interp)
        opti.set_initial(y, y_interp)

        # (Optional) Provide a reasonable initial guess
        # Use your RRT path for x,y and initialize T with its duration, then the solver is free to shrink it.
        T_init = float(path[-1][2] - path[0][2])
        opti.set_initial(T, T_init)

        # CasADi function to calculate detection probability of a path:
        # 4. Get the full symbolic decision variable vector (x, y, T)
        w_sym = ca.vertcat(x, y, T) 

        # 5. this calculates the log sum for a particular path
        log_sum_evaluator = ca.Function('physical_cost_evaluator', [w_sym], [log_sum])

        # 6. This calculates the time pnalty term for a particular path
        penalty_cost_evaluator = ca.Function('penalty_cost_evaluator', [w_sym], [time_penalty])

        # Calculating Probability of Detection for initial path:
        # Prepare the numerical initial guess vector (w0)
        w0 = ca.vertcat(x_interp, y_interp, T_init)

        # Evaluate the log sum for the initial path
        initial_log_sum_dm = log_sum_evaluator(w0)
        initial_log_sum_value = initial_log_sum_dm.full().item()

        # Evaluate the Penalty Cost for the initial path
        initial_penalty_cost_dm = penalty_cost_evaluator(w0)
        initial_penalty_cost_value = initial_penalty_cost_dm.full().item()

        # Calculate the inital probability of detection of the path:
        # Calculate and Report Results:
        initial_P_non_detection = np.exp(initial_log_sum_value)
        initial_P_detection = 1 - initial_P_non_detection
        initial_total_cost = initial_log_sum_value - initial_penalty_cost_value

        # Once you have the sum of the logarithms, here's how you convert back to a probability of detection:
        # log(1-k(x_1, t_1) + log(1-k(x_2, t_2) + log(1-k(x_3, t_3) + ... = log(P_not_detected)
        # exp(log(P_not_detected)) = P_not_detected
        # P_detected = 1 - P_not_detected
        # so P_detected = 1 - exp(log_sum)

        # Solver options
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 3000,
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
            "print_level": 0,
        }

        opti.solver("ipopt", p_opts, s_opts)

        # Solve
        try:
            sol = opti.solve()
            x_opt = sol.value(x)
            y_opt = sol.value(y)
            T_opt = float(sol.value(T))
            log_sum_opt = sol.value(log_sum)
        except RuntimeError as e:
            print("Solver failed:", e)
            x_opt = opti.debug.value(x)
            y_opt = opti.debug.value(y)
            T_opt = opti.debug.value(T)
            log_sum_opt = opti.debug.value(log_sum)

        # Calculating final probability of detection
        # Prepare the numerical final guess vector (wf)
        wf = ca.vertcat(x_opt, y_opt, T_opt)

        # Evaluate the log sum for the final path
        final_log_sum_dm = log_sum_evaluator(wf)
        final_log_sum_value = final_log_sum_dm.full().item()

        # Evaluate the Penalty Cost for the final path
        final_penalty_cost_dm = penalty_cost_evaluator(wf)
        final_penalty_cost_value = final_penalty_cost_dm.full().item()

        # Calculate the inital probability of detection of the path:
        # Calculate and Report Results:
        final_P_non_detection = np.exp(final_log_sum_value)
        final_P_detection = 1 - final_P_non_detection
        final_total_cost = final_log_sum_value - final_penalty_cost_value

        # Build a numeric time vector for export/animation
        t_opt_num = np.linspace(start_time, start_time + T_opt, N+1)

        # Final output: list of [x, y, t]
        optimized_path = list(zip(x_opt, y_opt, t_opt_num))

        # print(f"Final log_sum value: {final_log_sum_value:.4f}")
        # print(f"Final Time Penalty: {final_penalty_cost_value:.4f}")
        # print(f"Final Total Cost (J): {final_total_cost:.4f}")

        toc = time.time() - tic

        print(f"{initial_path_string} Probability of Detection: {initial_P_detection:.4f}")
        print(f"Optimized Probability of Detection: {final_P_detection:.4f}")
        print(f"Time to optimize path: {toc}")
        return initial_P_detection, final_P_detection, toc, optimized_path
    else:
        print(f"Scenario {scenario_key} does not have an initial path from {initial_path_string}")
        return None, None, None, None
    

# ParallelRRT_Pd_list = []
# opt_ParallelRRT_Pd_list = []
# opt_ParallelRRT_time_list = []

# RRTstar_Pd_list = []
# opt_RRTstar_Pd_list = []
# opt_RRTstar_time_list = []

# STRRT_Pd_list = []
# opt_STRRT_Pd_list = []
# opt_STRRT_time_list = []

# naive_Pd_list = []
# opt_naive_Pd_list = []
# opt_naive_time_list = []

# --- Replace the current list initializations with a single results dictionary ---
# This will store the final results:
# {
#   '0': { 'naive': {'Pd': 0.12, 'Time': 165.0}, 'RRTStar': {'Pd': 0.09, 'Time': 200.5}, ... },
#   '1': { ... }
# }
optimization_results = {}
# -------------------------------------------------------------------------------

# The list of initial path types to test
initial_path_types = ["naive", "RRTstar", "STRRT", "ParallelRRT"]

num_sims = 1 # 100 for real Monte Carlo sim

for i in range(num_sims):
    scenario_key = str(i)
    # Initialize the results for the current scenario if it doesn't exist
    if scenario_key not in optimization_results:
        optimization_results[scenario_key] = {}

    for path_type in initial_path_types:
        # Pass the scenario key and path type to the function
        initial_Pd, final_Pd, opt_time, opt_path = do_casadi_opt_sim(scenario_key, path_type)

        # Only store the result if the optimization was attempted (path was not None)
        if initial_Pd is not None:
            # Store the final optimized result using the scenario key and path type as sub-keys
            optimization_results[scenario_key][path_type] = {
                'initial_Pd': initial_Pd,
                'optimized_Pd': final_Pd,
                'opt_time': opt_time,
                'opt_path': opt_path
            }
        
    # Add a print statement to clearly separate runs for different path types
    print(f'End of scenario {i} calcs'+'-' * 40)
        
# --- End of loop ---

# Print the final structured results
#pprint.pprint(optimization_results)

# Save optimization results:
def save_optimization_results(data, filename="optimization_results.json"):
    """
    Saves a Python dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        filename (str): The name of the file to save the data to.
    """
    try:
        with open(filename, 'w') as f:
            # The 'indent=4' argument makes the JSON file human-readable
            # by formatting it with an indentation of 4 spaces.
            json.dump(data, f, indent=4)
        print(f"Successfully saved data to {filename}")
    except TypeError as e:
        print(f"Error saving to JSON: A TypeError occurred. This often means your dictionary contains objects (like numpy arrays or custom classes) that cannot be directly serialized to JSON. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")

# Example usage (assuming optimization_results is already populated)
save_optimization_results(optimization_results)


# List of all initial path types (must match the keys used in your simulation loop)
initial_path_types = ["naive", "RRTstar", "STRRT", "ParallelRRT"]

# 1. Initialize a dictionary to hold all collected values before averaging
all_values = {
    ptype: {'initial_Pd': [], 'optimized_Pd': [], 'opt_time': []} 
    for ptype in initial_path_types
}

# 2. Iterate through all scenario results and collect metrics by path type
for scenario_key, path_results in optimization_results.items():
    for path_type, metrics in path_results.items():
        if path_type in all_values:
            # Append all three metrics
            all_values[path_type]['initial_Pd'].append(metrics['initial_Pd'])
            all_values[path_type]['optimized_Pd'].append(metrics['optimized_Pd'])
            all_values[path_type]['opt_time'].append(metrics['opt_time'])



# Save all_values dictionary to a csv:
def save_full_results_to_csv(all_values, filename="full_optimization_runs.csv"):
    """
    Saves the full list of optimization results from all_values to a CSV
    with a multi-level header structure.
    
    The structure is:
    Top Row:    | naive |       |       | RRTstar |       |       | ...
    Second Row: | initial_Pd | optimized_Pd | opt_time | initial_Pd | optimized_Pd | opt_time | ...

    Args:
        all_values (dict): Dictionary containing lists of metrics per path type.
        filename (str): The name of the CSV file to save.
    """
    
    # 1. Prepare data for DataFrame creation
    data_for_df = {}
    path_types = list(all_values.keys())
    metric_headers = ['initial_Pd', 'optimized_Pd', 'opt_time']

    # 2. Create the MultiIndex Header and flatten the data
    
    # Generate a list of (Path_Type, Metric) tuples
    multi_index_tuples = list(itertools.product(path_types, metric_headers))
    
    # Create the pandas MultiIndex
    columns = pd.MultiIndex.from_tuples(multi_index_tuples, names=['Path_Type', 'Metric'])
    
    # Flatten the data structure: combine all lists into a single dictionary 
    # where the keys match the MultiIndex columns.
    flattened_data = {
        (ptype, metric): all_values[ptype][metric]
        for ptype in path_types
        for metric in metric_headers
    }
    
    # 3. Create the DataFrame using the flattened data
    # We use 'from_dict' with 'orient='index'' and then transpose (.T) 
    # to handle the lists of varying length and align data vertically.
    df = pd.DataFrame.from_dict(flattened_data, orient='index').T
    
    # 4. Apply the MultiIndex to the DataFrame columns
    df.columns = columns
    
    # 5. Save the DataFrame to a CSV file
    # We use header=False on the first save to handle the multi-level header 
    # manually, and then save the second row.
    try:
        # Save the top header row (Path_Type)
        # We manually flatten the index levels to get the desired header format
        # where the top level is saved, and the second level is appended.
        # This is a common workaround for MultiIndex CSV saving.
        df.to_csv(filename, header=True, index=False)
        print(f"Successfully saved full results to {filename}")

    except Exception as e:
        print(f"An error occurred while saving the CSV: {e}")

# --- 3. Execute the function ---
save_full_results_to_csv(all_values)




# 3. Calculate the final averages
average_results = {}
for path_type, metric_lists in all_values.items():
    # We can use any list here to check if data exists for this path_type
    if len(metric_lists['initial_Pd']) > 0:
        
        avg_initial_pd = np.mean(metric_lists['initial_Pd'])
        avg_optimized_pd = np.mean(metric_lists['optimized_Pd'])
        avg_time = np.mean(metric_lists['opt_time'])
        
        average_results[path_type] = {
            'N_runs': len(metric_lists['initial_Pd']),
            'Avg_Initial_Pd': avg_initial_pd,
            'Avg_Optimized_Pd': avg_optimized_pd,
            'Avg_Time_s': avg_time
        }

# 4. Convert to a DataFrame for presentation and saving
df_avg = pd.DataFrame.from_dict(average_results, orient='index')
df_avg.index.name = 'Path_Type'
df_avg = df_avg.reset_index()

# Sort by average optimized Pd (lowest is best)
df_avg_sorted = df_avg.sort_values(by='Avg_Optimized_Pd').round(4)

# Output the final results
csv_filename = 'full_average_optimization_results.csv'
df_avg_sorted.to_csv(csv_filename, index=False)