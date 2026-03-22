Hey Jefferson,

Here are the instructions for how to use this path planning optimizer code for the PURT demo

First of all, I've made minor tweaks in a BUNCH of random places in your existing code, so I just went ahead and included everything you should need in this folder. You'll want to save everything in the same folder and use that folder as your workspace in vscode (file > open folder > [place where you saved this])

Single_Scenario_Main (PURT 3_2) is the main Jupyter notebook that generates an STP-RRT* path and the CasADi optimized version of it. Here are the things you'll need to be able to change the scenario:

- ALL UNITS ARE IN METERS (not mm like in PURT). It was easier for the optimizer to have units that are close-ish in magnitude to 1.0.

- Change the map parameters in the third cell. Everything should be labeled decently well

- the panSpeed variable should allow you to switch the directions of the cameras. Just make them negative or positive. Double check that the animation from the python sim matches with the real-world panning (duh)

- The block "adding buildings" is where you put in building coordinates. Remember to convert to meters

- The RRT settings block allows you to change RRT things. They're decently dialed in for the PURT scenario right now, so I wouldn't mess with them unless you have to. 

--- ^^ The current vmax and max_time settings make it so that the uav will fly for 3 meters between waypoints. You can change max_time to make that shorter or longer. Make sure you don't accidentally make a path where the rrt and optimal paths go on opposite side of a building. Or Dr. Goppert will ask too many questions. If you really want to make sure that you can't do that, reduce max_time

- If you don't want to run an RRT when you run the code, just set run_RRT to False, and it will skip straight to optimization. The previous RRT will be loaded from the rrt_parameters.json file

- I think everything else should just work as is

- path_export_for_PURT_mm is the output file that I've been giving you for the demo runs. It has both the rrt and optimized paths converted to mm with the L-frame as the origin

ANIMATION INSTRUCTIONS:

Animation_Script (Obstacle Occlusion).py makes all of the nice animations and plots of the paths. MAKE SURE TO UPDATE THE FILE PATHS AT THE TOP OF THE SCRIPT. Otherwise you'll plot a different run of the optimization than you meant to.

- When you run the code, it will ask you to type the value of frame_skip that you want. Currently, the optimization saves the info at 100 Hz, so you don't want to animate every frame, or it will take forever to save. It usually takes a little less than 1 sec per frame for the animation to save, so choose wisely. The code will print out "DEBUG: Total interpolated points (N_common): ####" which tells you how many time steps there are from the optimization code. You'll want to skip some frames for sure. I usually do one run with frame_skip = 10000 to just check that the last frame looks good (saved as UAV_animation_final_frame). Then, I'll do frame_skip = 50 for checking how the path looks over time, then frame_skip = 10 if I want to make a really nice looking animation. 

I think that should be everything (and maybe more than you needed). Let me know if you have any questions. Thanks again for doing the experimental work! I hope PURT is nicer to you than it has been :P

