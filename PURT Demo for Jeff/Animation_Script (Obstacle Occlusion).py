import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import matplotlib.animation as animation
import json
import time
from scipy.interpolate import interp1d
from Camera import Camera
from Dynamic import DynamicMap
import matplotlib.lines as mlines
from matplotlib.path import Path
import os
print("Current working directory:", os.getcwd())

# ------------------------------------------------- Load variables from files -------------------------------------------------------


# Make sure to update these paths correctly
original_rrt_file_path = r"C:\Users\Joseph Kinerson\OneDrive - purdue.edu\Desktop\School Stuff\Research\Code\Purt Demo\Purt Demo for Jeff\rrt_original_path.csv"
opt_file_path = r"C:\Users\Joseph Kinerson\OneDrive - purdue.edu\Desktop\School Stuff\Research\Code\Purt Demo\Purt Demo for Jeff\optimized_path.csv"
sim_params_file_path = "C:/Users/Joseph Kinerson/OneDrive - purdue.edu/Desktop/School Stuff/Research/Code/Purt Demo/Purt Demo for Jeff/simulation_parameters.json"




# === Load path data from CSV files ===
def load_path_from_csv(filename):
    x, y, t = [], [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
            t.append(float(row['t']))
    return np.array(x), np.array(y), np.array(t)

def get_shadow_lines(camera_pos, obs_vertices):
    """
    Calculates the a, b, c coefficients for the lines bounding the shadow wedge.
    Returns a list of tuples: [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]
    """
    cx, cy = camera_pos
    
    # 1. Find the two tangent vertices (the visual silhouette)
    center_x = np.mean([v[0] for v in obs_vertices])
    center_y = np.mean([v[1] for v in obs_vertices])
    center_angle = math.atan2(center_y - cy, center_x - cx)
    
    angles = []
    for vx, vy in obs_vertices:
        ang = math.atan2(vy - cy, vx - cx)
        diff = (ang - center_angle + math.pi) % (2 * math.pi) - math.pi
        angles.append((diff, (vx, vy)))
        
    angles.sort(key=lambda item: item[0])
    v1 = angles[0][1]   # "Right" tangent vertex
    v2 = angles[-1][1]  # "Left" tangent vertex
    
    # 2. Define the three line segments
    line_segments = [
        (camera_pos, v1), # Right ray
        (camera_pos, v2), # Left ray
        (v1, v2)          # Front face
    ]
    
    # 3. Calculate a, b, c and Orient the Normals
    mid_x = (v1[0] + v2[0]) / 2.0
    mid_y = (v1[1] + v2[1]) / 2.0
    
    test_px = mid_x + (mid_x - cx)
    test_py = mid_y + (mid_y - cy)
    
    shadow_lines = []
    
    for p1, p2 in line_segments:
        x1, y1 = p1
        x2, y2 = p2
        
        a = y1 - y2
        b = x2 - x1
        c_val = x1*y2 - x2*y1
        
        test_val = a*test_px + b*test_py + c_val
        
        if test_val < 0:
            a, b, c_val = -a, -b, -c_val
            
        norm = math.hypot(a, b)
        if norm > 1e-6:
            a, b, c_val = a/norm, b/norm, c_val/norm
            
        shadow_lines.append((a, b, c_val))
        
    return shadow_lines


#x_rrt, y_rrt, t_rrt = load_path_from_csv('rrt_path.csv')
# original_rrt_file_path = r"C:\Users\Joseph Kinerson\OneDrive - purdue.edu\Desktop\School Stuff\Research\Code\Adding Obstacles\rrt_original_path.csv"
# opt_file_path = r"C:\Users\Joseph Kinerson\OneDrive - purdue.edu\Desktop\School Stuff\Research\Code\Adding Obstacles\optimized_path.csv"
# sim_params_file_path = "C:/Users/Joseph Kinerson/OneDrive - purdue.edu/Desktop/School Stuff/Research/Code/Adding Obstacles/simulation_parameters.json"

# Load STP-RRT path and Optimal Path:
x_rrt, y_rrt, t_rrt = load_path_from_csv(original_rrt_file_path)
x_opt, y_opt, t_opt = load_path_from_csv(opt_file_path)

# Load Defender and Map Parameters
with open(sim_params_file_path, "r") as f:
    params = json.load(f)
    
theta = params["theta"]
# c = params["c"]
# alpha = params["alpha"]
beta_distance = params["beta_distance"]
beta_pan = params["beta_pan"]
t_distance = params["t_distance"]
k_shadow = params["k_shadow"]
cameras_dict_copilot = params["cameras_dict_copilot"]
map_size = params["map_size"]
fov_range = params["fov_range"]
rrt_prob = params["initial_P_detection"]
opt_prob = params["final_P_detection"]
static_obstacles = params.get("static_obstacles", []) # <--- NEW ENTRY

cam_dict = {
    "n": len(params["cameras_dict_copilot"]),
    "x": [cam["pos"][0] for cam in params["cameras_dict_copilot"]],
    "y": [cam["pos"][1] for cam in params["cameras_dict_copilot"]],
    "spec": {
        "bound": [[cam["center"] - cam["amplitude"], cam["center"] + cam["amplitude"]] for cam in params["cameras_dict_copilot"]],
        "fov": [params["theta"], params["fov_range"]],
        "cam_time": [1 / params["cameras_dict_copilot"][0]["frequency"], 1],
        "init_angle": [cam["center"] for cam in params["cameras_dict_copilot"]],
        "panspeed": [cam.get("panspeed", 0.0) for cam in params["cameras_dict_copilot"]]
    }
}

camera_objects = [
    Camera(i, cam_dict, cam_dict["x"][i], cam_dict["y"][i])
    for i in range(cam_dict["n"])
]

# --- NEW: Pre-calculate obstacle occlusion lines for the animation ---
all_camera_shadows = []
# num_existing_obs = len(static_obstacles) - cam_dict["n"]  
# ^^ Assuming the first 'n' entries in static_obstacles are actually cameras, not real obstacles
# This prevents the camera bounding boxes from self-shadowing

for cam in camera_objects:
    camera_pos = cam.cam_position()
    camera_shadow_lines = []
    for obs_vertices in static_obstacles: # loop over all obstacles

        # NEW: Check if the camera is inside this building
        if Path(obs_vertices).contains_point(camera_pos):
            continue # Turn off shadow logic: skip this building entirely

        lines = get_shadow_lines(camera_pos, obs_vertices)
        camera_shadow_lines.append(lines)
    all_camera_shadows.append(camera_shadow_lines)
# ---------------------------------------------------------------------

# Create a common time base
common_start = min(t_rrt[0], t_opt[0])
common_end = max(t_rrt[-1], t_opt[-1])
# Force a reasonable resolution for the animation (e.g., 0.01s = 100 Hz)
# This prevents ultra-fine time steps from exploding the frame count.
common_dt = 0.01  
# Create the common time array
t_common = np.arange(common_start, common_end + common_dt, common_dt)

# common_dt = min(t_rrt[1] - t_rrt[0], t_opt[1] - t_opt[0])  # smallest dt for smoothness
# t_common = np.arange(common_start, common_end + common_dt, common_dt)

print(f"DEBUG: Total interpolated points (N_common): {len(t_common)}")



# Interpolate both paths onto the common time base
interp_rrt_x = interp1d(t_rrt, x_rrt, bounds_error=False, fill_value=(x_rrt[0], x_rrt[-1]))
interp_rrt_y = interp1d(t_rrt, y_rrt, bounds_error=False, fill_value=(y_rrt[0], y_rrt[-1]))
interp_opt_x = interp1d(t_opt, x_opt, bounds_error=False, fill_value=(x_opt[0], x_opt[-1]))
interp_opt_y = interp1d(t_opt, y_opt, bounds_error=False, fill_value=(y_opt[0], y_opt[-1]))

x_rrt_sync = interp_rrt_x(t_common)
y_rrt_sync = interp_rrt_y(t_common)
x_opt_sync = interp_opt_x(t_common)
y_opt_sync = interp_opt_y(t_common)

# ------------------------------------------------------ Cost Calculations --------------------------------------------------------------
# dt = t_common[1] - t_common[0]  # assuming uniform time step (and that dt is constant across different path types)

# # Updated version that uses calls to Camera.py to find the camera center
# def camera_cost_np(x, y, t, cam_obj):
#     x_c, y_c = cam_obj.cam_position()
#     phi = cam_obj.get_ctr_theta_t(t)
#     theta = cam_obj.fov_ang
#     dx = x - x_c
#     dy = y - y_c




#     distance_squared = dx**2 + dy**2
#     distance = np.sqrt(distance_squared + 1e-6)
#     cos_alpha = (dx * np.cos(phi) + dy * np.sin(phi)) / distance
#     visibility = 1 / (1 + np.exp(-k * (cos_alpha - np.cos(theta / 2))))
#     observability = 1 / (1 + alpha * distance_squared)
#     return visibility * observability

# def total_cost_np(x, y, t):
#     cost = np.zeros_like(x)
#     for cam_obj in camera_objects:
#         cost += camera_cost_np(x, y, t, cam_obj)
#     return cost




MAX_K_VALUE = 1



def calculate_k_value(xi, yi, ti, camera_objects, all_camera_shadows, beta_p, beta_d, t_d, k_shadow):
    '''
    This function calculates the k value at a given position (xi, yi) and time ti
    Contributions from multiple cameras should not be summed directly, since that can lead to k > 1. 
    Instead, we should calculate the probability of not being detected by each camera, 
    then multiply those together to get the total probability of not being detected, 
    and then take 1 minus that to get the total probability of being detected.
    
    k_shadow: Controls the sharpness of the shadow boundaries. 5.0 is a good starting point.
    '''
    p_not_detected = 1.0
    
    # Iterate through cameras and their corresponding pre-computed shadow lines
    for cam_idx, cam in enumerate(camera_objects):
        cam_x, cam_y = cam.cam_position()
        phi = cam.get_ctr_theta_t(ti)  # Use Camera.py method

        theta = cam.fov_ang
        dx = xi - cam_x
        dy = yi - cam_y
        dist2 = dx**2 + dy**2

        dist_min_sq = np.fmax(dist2, 1e-4) #try putting this back to 1e-4 if the optimization works
        dist_safe = np.sqrt(dist_min_sq) # "safe" because it prevents division by zero

        # cos_alpha = (dx * np.cos(phi) + dy * np.sin(phi)) / dist_safe
        
        # # Old visibility function
        # visibility = 1 / (1 + np.exp(-C_visibility * (cos_alpha - np.cos(theta / 2))))

        # # Old distance decay from Cartee (2019):
        # distance_decay = 1 / (1 + alpha * dist_min_sq)

        # ------------------------------------------------------------------
        # 1. NEW: Angular Visibility / Pan Angle Membership (Akbarzadeh et al., Eq 6)
        # ------------------------------------------------------------------
        # Calculate raw angle to target
        angle_to_target = np.atan2(dy, dx)
        
        # Calculate relative angle (gamma) between camera gaze and target
        gamma_raw = angle_to_target - phi
        
        # Normalize gamma to [-pi, pi] to satisfy the paper's range requirement
        gamma = np.atan2(np.sin(gamma_raw), np.cos(gamma_raw))
        
        # t_p controls the "width" of the function (half of the FOV)
        t_p = theta / 2.0
        
        # Double-sigmoid pan angle membership function
        vis_term1 = 1.0 / (1.0 + np.exp(-beta_p * (gamma + t_p)))
        vis_term2 = 1.0 / (1.0 + np.exp(-beta_p * (gamma - t_p)))
        visibility = vis_term1 - vis_term2

        # ------------------------------------------------------------------
        # 2. Distance decay (Akbarzadeh et al., Eq 5)
        # ------------------------------------------------------------------
        distance_decay = 1.0 - (1.0 / (1.0 + np.exp(-beta_d * (dist_safe - t_d))))
        
        # ------------------------------------------------------------------
        # Occlusion / Shadow Logic
        # ------------------------------------------------------------------
        # Start by assuming the robot is NOT occluded by any building
        total_vis_mult = 1.0 
        
        # all_camera_shadows[cam_idx] contains a list of buildings.
        # Each building is a list of 3 lines: (a, b, c)
        shadow_lines_for_this_cam = all_camera_shadows[cam_idx]
        
        for building_lines in shadow_lines_for_this_cam:
            # Assume it IS in the shadow of this specific building
            in_shadow_factor = 1.0
            
            for (a, b, c) in building_lines:

                # # 1. Calculate standard perpendicular distance (in meters)
                # perp_dist = a*xi + b*yi + c

                # Calculate signed distance to the boundary
                signed_dist = a*xi + b*yi + c
                
                # # 2. NEW: Convert to angular distance (in radians)
                # # By dividing by the distance to the camera
                # angular_dist = perp_dist / dist_safe


                # Smooth step: 1.0 if inside the boundary, 0.0 if outside
                smooth_bound = 0.5 * (np.tanh(k_shadow * signed_dist) + 1.0) # tanh version
                # smooth_bound = 1.0 / (1.0 + np.exp(-k_shadow * angular_dist))# sigmoid version, leads to overflow
                
                # Multiply: If the robot steps outside ANY of the 3 boundaries,
                # smooth_bound becomes 0, and in_shadow_factor drops to 0.
                in_shadow_factor *= smooth_bound
            
            # The visibility regarding THIS building is the inverse of being in its shadow
            building_vis_mult = 1.0 - in_shadow_factor
            
            # Multiply with the total visibility. 
            # If ANY building completely hides the robot, total_vis_mult becomes 0.
            total_vis_mult *= building_vis_mult
            
        # ------------------------------------------------------------------
        
        # 3. Combine everything
        # Actual probability of detection by this camera
        p_detect_actual = visibility * distance_decay * total_vis_mult
        
        # Update overall probability of not being detected by ANY camera
        p_not_detected *= (1 - p_detect_actual)
    
    return 1 - p_not_detected



# # New version that does obstacle occlusion
# def calculate_k_value(xi, yi, ti, camera_objects, all_camera_shadows, k_shadow=50):
#     p_not_detected = 1.0
    
#     for cam_idx, cam in enumerate(camera_objects):
#         cam_x, cam_y = cam.cam_position()
#         phi = cam.get_ctr_theta_t(ti)

#         theta = cam.fov_ang
#         dx = xi - cam_x
#         dy = yi - cam_y
#         dist2 = dx**2 + dy**2

#         dist_min_sq = np.fmax(dist2, 1e-4)
#         dist_safe = np.sqrt(dist_min_sq)

#         cos_alpha = (dx * np.cos(phi) + dy * np.sin(phi)) / dist_safe
        
#         # 1. Nominal FOV and Distance visibility
#         visibility = 1 / (1 + np.exp(-c * (cos_alpha - np.cos(theta / 2))))
#         distance_decay = 1 / (1 + alpha * dist_min_sq)
        
#         # 2. Occlusion / Shadow Logic
#         total_vis_mult = 1.0 
#         shadow_lines_for_this_cam = all_camera_shadows[cam_idx]
        
#         for building_lines in shadow_lines_for_this_cam:
#             in_shadow_factor = 1.0
            
#             for (a_line, b_line, c_line) in building_lines:
#                 # Perpendicular distance to the grid points
#                 perp_dist = a_line*xi + b_line*yi + c_line
                
#                 # # Convert to angular distance
#                 # angular_dist = perp_dist / dist_safe
                
#                 # # Old way with exp, leads to overflow errors
#                 # # Smooth step: 1.0 if inside boundary, 0.0 if outside
#                 # smooth_bound = 1.0 / (1.0 + np.exp(-k_shadow * angular_dist))

#                 # Smooth step: 1.0 if inside boundary, 0.0 if outside
#                 smooth_bound = 0.5 * (np.tanh(k_shadow * perp_dist) + 1.0)
#                 in_shadow_factor *= smooth_bound
            
#             building_vis_mult = 1.0 - in_shadow_factor
#             total_vis_mult *= building_vis_mult
            
#         # 3. Combine everything
#         p_detect_actual = visibility * distance_decay * total_vis_mult
#         p_not_detected *= (1 - p_detect_actual)
        
#     return 1 - p_not_detected



# # New version, uses call to Camera.py to find FOV center:
# def calculate_k_value(xi, yi, ti, camera_objects):
#     p_not_detected = 1
#     for cam in camera_objects:
#         cam_x, cam_y = cam.cam_position()
#         phi = cam.get_ctr_theta_t(ti)  # Use Camera.py method

#         theta = cam.fov_ang
#         dx = xi - cam_x
#         dy = yi - cam_y
#         dist2 = dx**2 + dy**2
#         # dist = np.sqrt(dist2 + 1e-6)

#         dist_min_sq = np.fmax(dist2, 1e-4) #try putting this back to 1e-4 if the optimization works
#         dist_safe = np.sqrt(dist_min_sq)

#         cos_alpha = (dx * np.cos(phi) + dy * np.sin(phi)) / dist_safe
#         visibility = 1 / (1 + np.exp(-c * (cos_alpha - np.cos(theta / 2))))
#         observability = 1 / (1 + alpha * dist_min_sq)
#         p_not_detected *= (1 - visibility * observability)
#     return 1 - p_not_detected 



# -------------------------------------------------------- ANIMATION CREATION -------------------------------------------
rrt_color = "#FF9D00"
opt_color = '#FFFFFF'
obstacle_color = "#FF0000FF"
camera_color = "#59FF00" 
start_color = "#00FFFB"
goal_color = "#F200FF"
building_alpha = 1

N_common = len(t_common)

# Grid for contour plot
x_vals = np.linspace(map_size[0], map_size[1], 200)
y_vals = np.linspace(map_size[2], map_size[3], 200)
X, Y = np.meshgrid(x_vals, y_vals)

# ... (Lines 150-160: X, Y meshgrid and initial contour setup)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(7, 7))
Z = calculate_k_value(X, Y, 0,camera_objects, all_camera_shadows,beta_pan, beta_distance, t_distance, k_shadow)
# Initial drawing of the cost landscape at t=0 (Plot this first, z-order=1)
contour = ax.contourf(X, Y, Z, levels=100, cmap='cividis', vmax = 1.0) # put vmax back to 0.1
cam_plot, = ax.plot([], [], "#FF00E1", label='Camera')



# 1. Create a "Proxy Artist" for the legend (represents all obstacles)
# We do this because the actual obstacles are individual polygons, but we only want one legend entry.
obstacle_proxy = patches.Patch(
    facecolor=obstacle_color, 
    edgecolor=obstacle_color, 
    alpha=building_alpha, 
    label='Obstacle'
)

# Create a proxy for the Camera (Magenta circle, no line)
camera_proxy = mlines.Line2D(
    [], [], 
    color=camera_color,    # Match the map color
    marker='o',         # Circle marker
    linestyle='None',   # REMOVE THE LINE
    markersize=8,       # Adjust size as needed
    label='Camera'
)

# Plot actual obstacles
for ii, vertices in enumerate(static_obstacles):
    polygon_patch = patches.Polygon(
        vertices,
        closed=True,
        facecolor=obstacle_color,
        edgecolor=obstacle_color,
        linewidth=1,
        alpha=building_alpha,
        zorder=3
    )
    ax.add_patch(polygon_patch)

# Plotting camera positions
for cam in cameras_dict_copilot:
    ax.plot(cam['pos'][0], cam['pos'][1], 'o', color=camera_color, markersize=4, zorder=4)

# 2. Capture Start and Goal handles using 'start_plot, =' syntax
start_plot, = ax.plot(x_rrt[0], y_rrt[0], 'x', color=start_color, markersize=12, zorder=4)
goal_plot, = ax.plot(x_rrt[-1], y_rrt[-1], 'x', color=goal_color, markersize=12, zorder=4)


# --- DYNAMIC OBJECTS (Initialize for update function) ---

rrt_dot, = ax.plot([], [], color=rrt_color, marker='o', markersize=6, zorder=5)
rrt_path_line, = ax.plot([], [], color=rrt_color, linestyle='dashed', linewidth=1, zorder=5)
opt_dot, = ax.plot([], [], color=opt_color, marker='o', markersize=6, zorder=5)
opt_path_line, = ax.plot([], [], color=opt_color, linestyle='--', linewidth=1, zorder=5)


# --- LEGEND & LAYOUT ---

title = ax.set_title(f'UAV Path Comparison, (t = 0.00s)')
plt.colorbar(contour, ax=ax, label='Single-Frame Probability of Detection, K')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.set_aspect('equal') # Forcing equal aspect ratio to prevent distortion of the map

# 3. Define the full list of handles and labels for the legend
legend_handles = [
    rrt_path_line, 
    opt_path_line,  
    camera_proxy, 
    obstacle_proxy,
    start_plot, 
    goal_plot
]

legend_labels = [
    f'STP-RRT*, P_d: {rrt_prob:.3f}', # Shortened for clarity
    f'Optimized, P_d: {opt_prob:.3f}', # Shortened for clarity
    'Camera', 
    'Obstacle',
    'Start', 
    'Goal'
]

# 4. Create and Style the Legend
leg = ax.legend(
    legend_handles,
    legend_labels,
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=3, 
    fancybox=True, 
    shadow=False
)

# Dark Background Styling
leg.get_frame().set_facecolor('#262626') 
leg.get_frame().set_edgecolor('black')
for text in leg.get_texts():
    text.set_color('white')

plt.subplots_adjust(bottom=0.25) # Increased bottom margin to fit the larger legend



# # --- STATIC OBJECTS (Plot next, Z-order > 1) ---

# obstacle_color = "#FF0000"
# building_alpha = 1

# # 1. Create a "Proxy Artist" for the legend (represents all obstacles)
# # We do this because the actual obstacles are individual polygons, but we only want one legend entry.
# obstacle_proxy = patches.Patch(
#     facecolor=obstacle_color, 
#     edgecolor=obstacle_color, 
#     alpha=building_alpha, 
#     label='Obstacle'
# )

# for ii, vertices in enumerate(static_obstacles):
#     polygon_patch = patches.Polygon(
#         vertices,
#         closed=True,
#         facecolor=obstacle_color,
#         edgecolor=obstacle_color,
#         linewidth=1,
#         alpha=building_alpha,
#         label=f'Obstacle {ii+1}' if ii == 0 else "",
#         zorder=3 # Ensure it's above the heatmap (zorder=1)
#     )
#     ax.add_patch(polygon_patch)

# # Plotting camera positions
# for cam in cameras_dict_copilot:
#     # Use a higher zorder to keep cameras visible
#     ax.plot(cam['pos'][0], cam['pos'][1], 'o', color="#FF00E1", zorder=4)

# # Add Start/Goal markers (assuming x_rrt[0], x_rrt[-1] are start/goal)
# ax.plot(x_rrt[0], y_rrt[0], 'x', color="#00FFFB", markersize=15, label='Start', zorder=4)
# ax.plot(x_rrt[-1], y_rrt[-1], 'x', color='#95FF00', markersize=15, label='Goal', zorder=4)


# # --- DYNAMIC OBJECTS (Initialize for update function) ---

# # plotting uav paths
# rrt_dot, = ax.plot([], [], color = 'orange', marker = 'o', markersize=6, label='STP-RRT* UAV', zorder=5)
# rrt_path_line, = ax.plot([], [], color = 'orange', linestyle = 'dashed', linewidth=1, label=f'STP-RRT*, P_d: {rrt_prob:.3f}', zorder=5)
# opt_dot, = ax.plot([], [], 'wo', markersize=6, label='Optimized UAV', zorder=5)
# opt_path_line, = ax.plot([], [], 'w--', linewidth=1, label=f'Optimized Path, P_d: {opt_prob:.3f}', zorder=5)

# # ... (rest of the code for titles, colorbar, legends, and the update function)


# # # Set up the figure and axis
# # fig, ax = plt.subplots(figsize=(8, 6))
# # Z = calculate_k_value(X, Y, 0,camera_objects)
# # contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
# # cam_plot, = ax.plot([], [], 'ro', label='Camera')

# # # Plotting static convex obstacles
# # obstacle_colors = ["#000000", "#636161"] # Gray colors for obstacles
# # building_alpha = 1

# # for ii, vertices in enumerate(static_obstacles):
# #     # vertices is a list of lists/tuples, which patches.Polygon accepts
# #     polygon_patch = patches.Polygon(
# #         vertices,
# #         closed=True,
# #         facecolor=obstacle_colors[ii % len(obstacle_colors)], # Cycle through colors
# #         edgecolor='k',
# #         linewidth=1,
# #         alpha=building_alpha,
# #         # Only label the first one for the legend
# #         label=f'Obstacle {ii+1}' if ii == 0 else "" 
# #     )
# #     ax.add_patch(polygon_patch)

# # # Plotting camera positions
# # for cam in cameras_dict_copilot:
# #     ax.plot(cam['pos'][0], cam['pos'][1], 'ro')

# # # plotting uav paths
# # rrt_dot, = ax.plot([], [], color = 'orange', marker = 'o', markersize=6)
# # rrt_path_line, = ax.plot([], [], color = 'orange', linestyle = 'dashed', linewidth=1)
# # opt_dot, = ax.plot([], [], 'wo', markersize=6)
# # opt_path_line, = ax.plot([], [], 'w--', linewidth=1)

# title = ax.set_title(f'UAV Path Comparison, (t = 0.00s)')
# plt.colorbar(contour, ax=ax, label='Single-Frame Probability of Detection, K')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.grid(True)

# # ... (previous lines defining title, colorbar, labels) ...
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.grid(True)

# # --- 1. Create the Legend ---
# # Save the legend object to a variable 'leg' so we can modify its style
# leg = ax.legend(
#     [cam_plot, rrt_path_line, opt_path_line],
#     ['Camera', f'STP-RRT*, P_d: {rrt_prob:.3f}', f'Optimized Path, P_d: {opt_prob:.3f}'],
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.15), # Places it below the plot
#     ncol=3
# )

# # --- 2. Style the Legend for Dark Mode ---
# # Set the background color of the legend box (dark gray)
# leg.get_frame().set_facecolor('#262626') 
# leg.get_frame().set_edgecolor('black')

# # Loop through the text items and set them to white
# for text in leg.get_texts():
#     text.set_color('white')

# # Adjust layout to make room for the legend at the bottom
# plt.subplots_adjust(bottom=0.2)

# # --- MODIFIED LEGEND ---
# # loc='upper center' combined with bbox_to_anchor anchors the legend relative to the plot
# # (0.5, -0.15) places it centered horizontally (0.5) and 15% below the bottom axis (-0.15)
# # ncol=3 spreads the items out horizontally so it doesn't take up too much vertical space
# ax.legend([cam_plot, rrt_path_line, opt_path_line], 
#           ['Camera', f'STP-RRT*, P_d: {rrt_prob:.3f}', f'Optimized Path, P_d: {opt_prob:.3f}'], 
#           loc='upper center', 
#           bbox_to_anchor=(0.5, -0.15), 
#           ncol=3, 
#           fancybox=True, 
#           shadow=False)

# # Adjust layout to ensure the new legend isn't cut off when saving
# plt.subplots_adjust(bottom=0.2)

# title = ax.set_title(f'UAV Path Comparison, (t = 0.00s)')
# plt.colorbar(contour, ax=ax, label='Single-Frame Probability of Detection, K')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.grid(True)
# ax.legend([cam_plot, rrt_path_line, opt_path_line], ['Camera', f'STP-RRT*, P_d: {rrt_prob:.3f}',\
#                                                  f'Optimized Path, P_d: {opt_prob:.3f}'], loc = 'upper right')



# Uncomment for plotting only the optimal path
#ax.legend([cam_plot, opt_path_line], ['Camera', f'Optimized Path, P_d: {opt_prob:.3f}'], loc = 'upper left')


# Defining start and end
start_frame = 0
# start_frame = int(np.ceil(9 * N_common / 10)) # Uncomment this if you just want the last few frames
end_frame = N_common - 1

# Updated version prompts the user for the frame_skip value
try:
    user_input = input("Enter frame_skip (Press Enter for default 100): ")
    if user_input.strip() == "":
        frame_skip = 100
    else:
        frame_skip = int(user_input)
except ValueError:
    print("Invalid input. Defaulting to 100.")
    frame_skip = 100

print(f"Frame skip set to {frame_skip} for animation.")


max_frames = max(len(t_rrt), len(t_opt))


frame_indices = list(range(start_frame, end_frame, frame_skip))
if frame_indices[-1] != end_frame:
    frame_indices.append(end_frame)

anim_time = t_common[end_frame] - t_common[start_frame]  
interval = common_dt* 1000  # milliseconds per frame

tic = time.time()

def update(frame):
    t = t_common[frame]

    # Clear previous contour
    for c in ax.collections:
        c.remove()
    Z = calculate_k_value(X, Y, t,camera_objects, all_camera_shadows, beta_pan, beta_distance, t_distance, k_shadow)
    ax.contourf(X, Y, Z, levels=100, cmap='cividis', vmax = 1.0) # put vmax back to 0.1

    # # Remove previous FOV triangles
    # [patch.remove() for patch in ax.patches]

    # # Drawing FOV triangles with camera.get_fov() call
    # for cam in camera_objects:
    #     fov_triangle = cam.get_fov(cam.x0, cam.y0, t)
    #     triangle = patches.Polygon(fov_triangle, closed=True, color='red', alpha=0.2)
    #     ax.add_patch(triangle)

    # Update UAV paths
    rrt_path_x = x_rrt_sync[:frame + 1]
    rrt_path_y = y_rrt_sync[:frame + 1]
    rrt_path_line.set_data(rrt_path_x, rrt_path_y)
    rrt_dot.set_data([x_rrt_sync[frame]], [y_rrt_sync[frame]])

    opt_path_x = x_opt_sync[:frame + 1]
    opt_path_y = y_opt_sync[:frame + 1]
    opt_path_line.set_data(opt_path_x, opt_path_y)
    opt_dot.set_data([x_opt_sync[frame]], [y_opt_sync[frame]])

    # title.set_text(f'UAV Path Comparison, (t = {t:.2f}s)')
    title.set_text(f'Optimized Path from STP-RRT*, (t = {t:.2f}s)')
    return [rrt_path_line, rrt_dot, opt_path_line, opt_dot, title]
    #return [opt_path_line, opt_dot, title] # if you only want the optimal path plotted

# Create animation
if x_opt is None or y_opt is None or len(x_opt) == 0 or len(y_opt) == 0:
    raise ValueError("Optimization failed or returned empty path. Cannot animate.")



frame_indices = list(range(start_frame, end_frame, frame_skip)) 
if frame_indices[-1] != end_frame:
    frame_indices.append(end_frame)

# anim_time = t_opt[end_frame] - t_opt[start_frame]  
# interval = dt* 1000  # milliseconds per frame

total_duration = max(t_rrt[-1], t_opt[-1])
num_frames = len(frame_indices)
interval = (total_duration / num_frames) * 1000  # milliseconds per frame


tic = time.time()

# ----------------------------------------------------------------------------
# START OF MODIFIED/ADDED CODE (Inserting after Line 462)
# ----------------------------------------------------------------------------

# 1. Update the plot to the final frame state for PNG save
final_frame_index = N_common - 1 
update(final_frame_index)

# 2. Save the final state as a PNG
png_save_name = 'UAV_animation_final_frame.png'
plt.savefig(png_save_name, dpi=600)
print(f"Final frame saved as {png_save_name}.")


# --- REPLACEMENT FOR LINES 475-479 ---

# 1. Calculate the exact duration of the simulation segment being animated
sim_duration = t_common[end_frame] - t_common[start_frame]

# 2. Count exactly how many frames we are about to render
actual_frame_count = len(frame_indices)

# 3. Force the FPS to match real-time playback
# (Frames / Seconds = FPS)
real_time_fps = actual_frame_count / sim_duration

print(f"--- ANIMATION PARAMETERS ---")
print(f"Simulation Duration: {sim_duration:.2f} seconds")
print(f"Frames to Render:    {actual_frame_count}")
print(f"Calculated FPS:      {real_time_fps:.2f}")
print(f"--------------------------")


# 3. Create the animation object using the full list of frames for the GIF
# This replaces the original single-frame anim call (which was commented out or used a single frame)
anim = animation.FuncAnimation(fig, update, frames=frame_indices, interval=interval, blit=False)


# # Save animation as MP4 using ffmpeg
# effective_fps = (1/dt)/frame_skip

save_name = 'UAV_animation.gif'
anim.save(save_name, fps=real_time_fps, dpi=300, writer='Pillow')


# save_name = 'UAV_animation.gif'

# anim.save(save_name, fps=effective_fps, dpi=300, writer='Pillow')

elapsed = time.time() - tic

print(f"Animation saved as {save_name}.")
print(f'Time to render and save animation: {elapsed:.2f} sec')