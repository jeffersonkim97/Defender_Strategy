import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import matplotlib.animation as animation
import json
import time
from scipy.interpolate import interp1d
from Camera import Camera
from Dynamic import DynamicMap
# import os
# print("Current working directory:", os.getcwd())

# ------------------------------------------------- Load variables from files -------------------------------------------------------

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

#x_rrt, y_rrt, t_rrt = load_path_from_csv('rrt_path.csv')
original_rrt_file_path = r"C:\Users\Joseph Kinerson\OneDrive - purdue.edu\Desktop\School Stuff\Research\Code\Monte Carlo\rrt_original_path.csv"
opt_file_path = r"C:\Users\Joseph Kinerson\OneDrive - purdue.edu\Desktop\School Stuff\Research\Code\Monte Carlo\optimized_path.csv"
sim_params_file_path = "C:/Users/Joseph Kinerson/OneDrive - purdue.edu/Desktop/School Stuff/Research/Code/Monte Carlo/simulation_parameters.json"

# Load STP-RRT path and Optimal Path:
x_rrt, y_rrt, t_rrt = load_path_from_csv(original_rrt_file_path)
x_opt, y_opt, t_opt = load_path_from_csv(opt_file_path)

# Load Camera Parameters
with open(sim_params_file_path, "r") as f:
    params = json.load(f)
    
theta = params["theta"]
k = params["k"]
alpha = params["alpha"]
cameras_dict_copilot = params["cameras_dict_copilot"]
map_size = params["map_size"]
fov_range = params["fov_range"]
rrt_prob = params["initial_P_detection"]
opt_prob = params["final_P_detection"]

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

# Create a common time base
common_start = min(t_rrt[0], t_opt[0])
common_end = max(t_rrt[-1], t_opt[-1])
common_dt = min(t_rrt[1] - t_rrt[0], t_opt[1] - t_opt[0])  # smallest dt for smoothness
t_common = np.arange(common_start, common_end + common_dt, common_dt)

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
dt = t_common[1] - t_common[0]  # assuming uniform time step (and that dt is constant across different path types)

# Updated version that uses calls to Camera.py to find the camera center
def camera_cost_np(x, y, t, cam_obj):
    x_c, y_c = cam_obj.cam_position()
    phi = cam_obj.get_ctr_theta_t(t)
    theta = cam_obj.fov_ang
    dx = x - x_c
    dy = y - y_c
    distance_squared = dx**2 + dy**2
    distance = np.sqrt(distance_squared + 1e-6)
    cos_alpha = (dx * np.cos(phi) + dy * np.sin(phi)) / distance
    visibility = 1 / (1 + np.exp(-k * (cos_alpha - np.cos(theta / 2))))
    observability = 1 / (1 + alpha * distance_squared)
    return visibility * observability

def total_cost_np(x, y, t):
    cost = np.zeros_like(x)
    for cam_obj in camera_objects:
        cost += camera_cost_np(x, y, t, cam_obj)
    return cost



# -------------------------------------------------------- ANIMATION CREATION -------------------------------------------

N_common = len(t_common)

# Grid for contour plot
x_vals = np.linspace(map_size[0], map_size[1], 200)
y_vals = np.linspace(map_size[2], map_size[3], 200)
X, Y = np.meshgrid(x_vals, y_vals)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
Z = total_cost_np(X, Y, 0)
contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
cam_plot, = ax.plot([], [], 'ro', label='Camera')

# Plotting camera positions
for cam in cameras_dict_copilot:
    ax.plot(cam['pos'][0], cam['pos'][1], 'ro')

# plotting uav paths
rrt_dot, = ax.plot([], [], color = 'orange', marker = 'o', markersize=6)
rrt_path_line, = ax.plot([], [], color = 'orange', linestyle = 'dashed', linewidth=1)
opt_dot, = ax.plot([], [], 'wo', markersize=6)
opt_path_line, = ax.plot([], [], 'w--', linewidth=1)

title = ax.set_title(f'UAV Path Comparison, (t = 0.00s)')
plt.colorbar(contour, ax=ax, label='Pointwise Probability of Detection, K')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.legend([cam_plot, rrt_path_line, opt_path_line], ['Camera', f'STP-RRT*, P_detected: {rrt_prob:.3f}',\
                                                 f'Optimized Path, P_detected: {opt_prob:.3f}'], loc = 'upper left')

ax.legend([cam_plot, opt_path_line], ['Camera', f'Optimized Path, P_detected: {opt_prob:.3f}'], loc = 'upper left')


# Defining start and end
start_frame = 0 # int(np.ceil(9 * N_common / 10)) # Uncomment this if you just want the last few frames
end_frame = N_common - 1

# Choose based on quality and time to save the gif: FPS calcs are wrong
# frame_skip = 2 # 30 FPS, 11 minutes to save
# frame_skip = 4 # 15 FPS,  3 min 40 sec to save
# frame_skip = 8 # 7.5 FPS, ~ 2 min to save
frame_skip = 16 # 3.25 FPS, 40 sec to save

max_frames = max(len(t_rrt), len(t_opt))


frame_indices = list(range(start_frame, end_frame, frame_skip))
if frame_indices[-1] != end_frame:
    frame_indices.append(end_frame)

anim_time = t_common[end_frame] - t_common[start_frame]  
interval = dt* 1000  # milliseconds per frame

tic = time.time()

# Save animation as MP4 using ffmpeg
effective_fps = (1/dt)/frame_skip


def update(frame):
    t = t_common[frame]

    # Clear previous contour
    for c in ax.collections:
        c.remove()
    Z = total_cost_np(X, Y, t)
    ax.contourf(X, Y, Z, levels=100, cmap='viridis', vmax = 0.1)

    # # Remove previous FOV triangles
    # [patch.remove() for patch in ax.patches]

    # # Drawing FOV triangles with camera.get_fov() call
    # for cam in camera_objects:
    #     fov_triangle = cam.get_fov(cam.x0, cam.y0, t)
    #     triangle = patches.Polygon(fov_triangle, closed=True, color='red', alpha=0.2)
    #     ax.add_patch(triangle)

    # # Update UAV paths
    # rrt_path_x = x_rrt_sync[:frame + 1]
    # rrt_path_y = y_rrt_sync[:frame + 1]
    # rrt_path_line.set_data(rrt_path_x, rrt_path_y)
    # rrt_dot.set_data([x_rrt_sync[frame]], [y_rrt_sync[frame]])

    opt_path_x = x_opt_sync[:frame + 1]
    opt_path_y = y_opt_sync[:frame + 1]
    opt_path_line.set_data(opt_path_x, opt_path_y)
    opt_dot.set_data([x_opt_sync[frame]], [y_opt_sync[frame]])

    # title.set_text(f'UAV Path Comparison, (t = {t:.2f}s)')
    title.set_text(f'Optimized Path, Time Penalty = 0.02*T, (t = {t:.2f}s)')
    #return [rrt_path_line, rrt_dot, opt_path_line, opt_dot, title]
    return [opt_path_line, opt_dot, title] # if you only want the optimal path plotted

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
plt.savefig(png_save_name, dpi=300)
print(f"Final frame saved as {png_save_name}.")

# 3. Create the animation object using the full list of frames for the GIF
# This replaces the original single-frame anim call (which was commented out or used a single frame)
anim = animation.FuncAnimation(fig, update, frames=frame_indices, interval=interval, blit=False)


# Save animation as MP4 using ffmpeg
effective_fps = (1/dt)/frame_skip

save_name = 'UAV_animation.gif'

anim.save(save_name, fps=effective_fps, dpi=300, writer='Pillow')

elapsed = time.time() - tic

print(f"Animation saved as {save_name}.")
print(f'Time to render and save animation: {elapsed:.2f} sec')




# ================================== RANDOM OLD CODE: ==============================================================

# Original version (only making the optimal path animation)

# #def update(frame):
#     t = t_opt[frame]
#     Z = total_cost_np(X, Y, t)

#     # Clear previous contour
#     for c in ax.collections:
#         c.remove()

#     # Draw new contour
#     ax.contourf(X, Y, Z, levels=100, cmap='viridis')

#     # # Replot cameras
#     # for cam in cameras_dict_copilot:
#     #     ax.plot(cam['pos'][0], cam['pos'][1], 'ro')

#     # Remove previous FOV triangles
#     [patch.remove() for patch in ax.patches]

#     for cam in cameras_dict_copilot:
#         x_c, y_c = cam['pos']
#         amplitude = cam['amplitude']
#         frequency = cam['frequency']
#         # phase = cam['phase']
#         phase = 0 # Assuming all cameras start at center of angular range
#         center = cam['center']
#         fov_angle = theta  # total FOV angle in radians
#         fov_range = 5      # how far the camera can see

#         # Compute current pan angle
#         phi = center + amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
#         # phi_ccw = center + amplitude * np.sin(2 * np.pi * frequency * t + phase)
#         # phi = (2 * np.pi - phi_ccw) % (2 * np.pi) # adjusting to make positive angles clockwise

#         # # Compute left and right edge angles
#         # left_angle = phi - fov_angle / 2
#         # right_angle = phi + fov_angle / 2

#         # # Compute triangle points
#         # left_point = (x_c + fov_range * np.cos(left_angle), y_c + fov_range * np.sin(left_angle))
#         # right_point = (x_c + fov_range * np.cos(right_angle), y_c + fov_range * np.sin(right_angle))

#         # # Create triangle patch
#         # triangle = patches.Polygon(
#         #     [[x_c, y_c], left_point, right_point],
#         #     closed=True,
#         #     color='red',
#         #     alpha=0.2
#         # )
#         # ax.add_patch(triangle)

#         # Draw camera position
#         ax.plot(x_c, y_c, 'ro')

#     # Update UAV path and dot
#     rrt_path_x = x_rrt[:frame+1]
#     rrt_path_y = y_rrt[:frame+1]
#     rrt_path_line.set_data(rrt_path_x, rrt_path_y)
#     rrt_dot.set_data([x_rrt[frame]], [y_rrt[frame]])

#     # opt_path_x = x_opt[:frame+1]
#     # opt_path_y = y_opt[:frame+1]
#     # opt_path_line.set_data(opt_path_x, opt_path_y)
#     # opt_dot.set_data([x_opt[frame]], [y_opt[frame]])

#     opt_path_x = x_opt_full[:frame+1]
#     opt_path_y = y_opt_full[:frame+1]
#     opt_path_line.set_data(opt_path_x, opt_path_y)
#     opt_dot.set_data([x_opt_full[frame]], [y_opt_full[frame]])

 
#     title.set_text(f'UAV Path Comparison, (t = {t:.2f}s)')

#     # Return only the updated artists
#     return [rrt_path_line, rrt_dot, opt_path_line, opt_dot, title]

# def update(frame):
#     if frame < len(t_rrt):
#         rrt_path_x = x_rrt[:frame+1]
#         rrt_path_y = y_rrt[:frame+1]
#         rrt_path_line.set_data(rrt_path_x, rrt_path_y)
#         rrt_dot.set_data([x_rrt[frame]], [y_rrt[frame]])
#         t_rrt_frame = t_rrt[frame]
#     else:
#         t_rrt_frame = t_rrt[-1]

#     if frame < len(t_opt):
#         opt_path_x = x_opt[:frame+1]
#         opt_path_y = y_opt[:frame+1]
#         opt_path_line.set_data(opt_path_x, opt_path_y)
#         opt_dot.set_data([x_opt[frame]], [y_opt[frame]])
#         t_opt_frame = t_opt[frame]
#     else:
#         t_opt_frame = t_opt[-1]

#     # Use the maximum of the two times for cost visualization
#     t = max(t_rrt_frame, t_opt_frame)

#     Z = total_cost_np(X, Y, t)
#     for c in ax.collections:
#         c.remove()
#     ax.contourf(X, Y, Z, levels=100, cmap='viridis')
#     title.set_text(f'UAV Path Comparison, (t = {t:.2f}s)')
#     return [rrt_path_line, rrt_dot, opt_path_line, opt_dot, title]

# 



    # # Draw FOV triangles for each camera
    # for cam in cameras_dict_copilot:
    #     x_c, y_c = cam['pos']
    #     amplitude = cam['amplitude']
    #     frequency = cam['frequency']
    #     phase = cam['phase']
    #     center = cam['center']
    #     phi = center + amplitude * np.sin(2 * np.pi * frequency * t + phase)

    #     left_angle = phi - theta / 2
    #     right_angle = phi + theta / 2

    #     left_point = (x_c + fov_range * np.cos(left_angle), y_c + fov_range * np.sin(left_angle))
    #     right_point = (x_c + fov_range * np.cos(right_angle), y_c + fov_range * np.sin(right_angle))

    #     triangle = patches.Polygon(
    #         [[x_c, y_c], left_point, right_point],
    #         closed=True,
    #         color='red',
    #         alpha=0.2
    #     )
    #     ax.add_patch(triangle)



    
# Old version that just calculates the camera headings with the sinusoidal function
# # === Camera cost function ===
# def camera_cost_np(x, y, t, cam):
#     x_c, y_c = cam['pos']
#     phi = cam['center'] + cam['amplitude'] * np.sin(2 * np.pi * cam['frequency'] * t + cam['phase'])
#     dx = x - x_c
#     dy = y - y_c
#     distance_squared = dx**2 + dy**2
#     distance = np.sqrt(distance_squared)
#     distance = np.where(distance == 0, 1e-6, distance)
#     cos_alpha = (dx * np.cos(phi) + dy * np.sin(phi)) / distance
#     visibility = 1 / (1 + np.exp(-k * (cos_alpha - np.cos(theta / 2))))
#     observability = 1 / (1 + alpha * distance_squared)
#     return visibility * observability

# # Total cost function
# def total_cost_np(x, y, t):
#     cost = np.zeros_like(x)
#     for cam in cameras_dict_copilot:
#         cost += camera_cost_np(x, y, t, cam)
#     return cost

# # === Exposure and detection probability ===
# def compute_exposure_and_probability(x_path, y_path, t_path):
#     exposure_cost = 0
#     for i in range(len(x_path) - 1):
#         xm = 0.5 * (x_path[i] + x_path[i+1])
#         ym = 0.5 * (y_path[i] + y_path[i+1])
#         tm = 0.5 * (t_path[i] + t_path[i+1])
#         for cam in cameras_dict_copilot:
#             exposure_cost += camera_cost_np(xm, ym, tm, cam) * dt
#     P_detected = 1 - np.exp(-exposure_cost)
#     return exposure_cost, P_detected


# # creating camera dictionary
# camera_objects = []
# for i, cam in enumerate(params["cameras_dict_copilot"]):
#     cam_spec = {
#         "spec": {
#             "bound": [[cam["center"] - cam["amplitude"], cam["center"] + cam["amplitude"]]],
#             "fov": [params["theta"], cam["range"]],
#             "cam_time": [1 / cam["frequency"], 1],  # period and dt
#             "init_angle": [cam["center"]],
#             "panspeed": [cam["panspeed"]]  
#         }
#     }
#     cam_obj = Camera(i, cam_spec, cam["pos"][0], cam["pos"][1])
#     camera_objects.append(cam_obj)

# creating camera dictionary
# cam_dict = {
#     "n": len(cameras_dict_copilot),
#     "x": [cam["pos"][0] for cam in cameras_dict_copilot],
#     "y": [cam["pos"][1] for cam in cameras_dict_copilot],
#     "spec": {
#         "bound": [[cam["center"] - cam["amplitude"], cam["center"] + cam["amplitude"]] for cam in cameras_dict_copilot],
#         "fov": [params["theta"], params["fov_range"]],
#         "cam_time": [1 / cameras_dict_copilot[0]["frequency"], 1],
#         "init_angle": [cam["center"] for cam in cameras_dict_copilot],
#         "panspeed": [cam["panspeed"] for cam in cameras_dict_copilot]
#     }
# }
# camera_objects = [
#     Camera(i, cam_dict, cam_dict["x"][i], cam_dict["y"][i])
#     for i in range(cam_dict["n"])
# ]



# # === Exposure and detection probability ===
# # THIS IS GETTING CUT FROM NEW VERSIONS: SWITCHING TO LOG TRANSFORMATION IN MAIN SCRIPT
# def compute_exposure_and_probability(x_path, y_path, t_path):
#     exposure_cost = 0
#     for i in range(len(x_path) - 1):
#         xm = 0.5 * (x_path[i] + x_path[i+1])
#         ym = 0.5 * (y_path[i] + y_path[i+1])
#         tm = 0.5 * (t_path[i] + t_path[i+1])
#         for cam_obj in camera_objects:
#             exposure_cost += camera_cost_np(xm, ym, tm, cam_obj) * dt
#     P_detected = 1 - np.exp(-exposure_cost)
#     return exposure_cost, P_detected
# # Calculate exposure integral and probability of detection
# rrt_exposure, rrt_prob = compute_exposure_and_probability(x_rrt, y_rrt, t_rrt)
# opt_exposure, opt_prob = compute_exposure_and_probability(x_opt, y_opt, t_opt)