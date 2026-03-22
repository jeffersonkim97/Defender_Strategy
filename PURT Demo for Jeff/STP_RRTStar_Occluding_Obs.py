
# Required Python Packages
import numpy as np
import random as rn
import time
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from Dynamic import DynamicMap
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial import KDTree
from matplotlib.path import Path
import math

import Dynamic
from scipy.interpolate import splprep, splev
import importlib
import csv
import json
from scipy.interpolate import interp1d
from Camera import Camera
import time
import convex_map_PURT as convex_map
from typing import Dict, List, Tuple, Optional
import random
import math
from matplotlib.path import Path
from shapely.geometry import MultiPoint, Polygon, MultiPolygon, GeometryCollection

from Dynamic import DynamicMap



class STP_RRTStar_Occluding_Obs():

    def __init__(self, vmax, x0, xf, map_size, vehicle, cam_dict, dmap, max_time, map_in, proximity):
        self.vmax = vmax
        self.x0 = x0
        self.xf = xf
        self.map_size = map_size
        self.vehicle = vehicle 
        self.cam_dict = cam_dict # dictionary of cameras
        self.dmap = dmap # dynamic map object
        self.max_time = max_time
        self.map_in = map_in
        self.proximity_space = proximity[0]
        self.proximity_time = proximity[1]

        # --- NEW: Pre-calculate shadows for RRT ---
        self.all_camera_shadows = []
        
        for cam_idx in range(self.cam_dict['n']):
            camera_pos = (self.cam_dict['x'][cam_idx], self.cam_dict['y'][cam_idx])
            camera_shadow_lines = []
            
            for obs_idx in range(self.map_in['st']['n']):
                obs_vertices = self.map_in['st'][str(obs_idx)]
                
                # Check if camera is inside building, if so, turn off shadow logic for it
                if Path(obs_vertices).contains_point(camera_pos):
                    continue
                    
                lines = self.get_shadow_lines(camera_pos, obs_vertices)
                camera_shadow_lines.append(lines)
                
            self.all_camera_shadows.append(camera_shadow_lines)
        # ------------------------------------------

        print('Occluding Obstacle STP-RRTStar 2/20/26 9:03 PM version initialized')

    def get_shadow_lines(self, camera_pos, obs_vertices):
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

    # ------------------------- STP-RRT* Subfunctions ----------------------------------------------#
    def distance(self, a, b):
        """
        Euclidean distance between two points
        """
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def random_sample_test2(self, x0, tf, nn, k):
        """
        Generate convex hull within space-time domain and sample random node in uniform random within convex hull
        """
        trand = rn.uniform(0, tf)
        sample_goal = rn.uniform(0, 1) < 0.1
        # Find a random point in a domain
        if sample_goal:
            # IF not, sample goal
            if np.mod(k, nn) == 0:
                xrand = self.xf[0]
                yrand = self.xf[1]
                trand = tf
            else:
                xrand = x0[0]
                yrand = x0[1]
                trand = x0[2]
        elif not sample_goal:
            # Construct Points for Convex Hull
            point1 = [x0[0], x0[2]]
            point2 = [0, x0[0]/self.vmax]
            point3 = [self.map_size[0], (self.map_size[0]-x0[0])/self.vmax]
            point4 = [0, tf-self.xf[0]/self.vmax]
            point5 = [self.map_size[0], tf+self.xf[0]/self.vmax-self.map_size[0]/self.vmax]
            point6 = [self.xf[0], tf]
            pos = [point1, point2, point3, point4, point5, point6]

            # Convex hull
            hull = ConvexHull( pos )
            # Bounding box
            bbox = [hull.min_bound, hull.max_bound]
            #Hull path
            hull_path = Path(hull.points[hull.vertices])
            # Draw n
            rand_points = np.empty((1, 2))
            for i in range(1):
                rand_points[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])

                while hull_path.contains_point(rand_points[i]) == False:
                    rand_points[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
            xrand = rand_points[0][0]
            trand = rand_points[0][1]
            yrand = rn.uniform(np.min([0, x0[0]/self.vmax]), np.min([self.vmax*(trand+x0[0]/self.vmax), self.map_size[3]]))

        return [xrand, yrand, trand]

    # def reachable(self, q0, q1, forward=True):
    #     """
    #     Check if q1 is within reachable set from q0
    #     """
    #     dx = np.abs(q1[0]-q0[0])
    #     dy = np.abs(q1[1]-q0[1])
    #     dd = np.sqrt(dx**2 + dy**2)
    #     dt = q1[2]-q0[2]

    #     if dd == 0:
    #         return True
    #     else:
    #         if forward and dt > 0 and dd/dt <= self.vmax:
    #             return True
    #         elif not forward and dt < 0 and dd/dt <= self.vmax:
    #             return True
    #         return False


    # Updated version to ensure that the speed constraint is enforced
    # for both the forward and backward trees
    def reachable(self, q0, q1, forward=True):
        """
        Check if q1 is within reachable set from q0
        """
        dx = np.abs(q1[0]-q0[0])
        dy = np.abs(q1[1]-q0[1])
        dd = np.sqrt(dx**2 + dy**2)
        dt = q1[2]-q0[2]

        # Prevent division by zero
        if abs(dt) < 1e-9:
            # If distance is zero, we are there. If distance > 0 and time=0, it's impossible.
            return dd == 0

        # Check time direction constraints
        if forward and dt <= 0:
            return False  # Start tree must move forward in time
        if not forward and dt >= 0:
            return False  # Goal tree must move backward in time

        # Check Speed Constraint: speed = distance / |time|
        speed = dd / abs(dt)
        
        if speed <= self.vmax:
            return True
            
        return False
        
    def find_neighbor(self, qrand, V, n, k, query_max=10, closest=True):
        """
        Find closest neighbor in list of vertex V from qrand
        """
        # Find Neighbor
        if np.mod(k,n) == 0:
            forward = True
        else:
            forward = False

        kdtree=KDTree(V)
        dist,points=kdtree.query(qrand,query_max)
        points_list = [i for n, i in enumerate(points) if i not in points[:n]]

        if closest:
            if len(V) < query_max:
                for ii in range(len(V)):
                    pointToCheck = V[points_list[ii]]
                    if self.reachable(pointToCheck, qrand, forward):
                        return pointToCheck
                return None
            else:
                for ii in range(len(points_list)):
                    pointToCheck = V[points_list[ii]]
                    if self.reachable(pointToCheck, qrand, forward):
                        return pointToCheck
                return None
        else:
            return points
        
    def extend(self, q0, q1, max_time, forward=True):
        """
        Extend towards q1. 
        - Logic: Fly at V_MAX unless we are close enough (in time and space) 
          to hit the target exactly within this step.
        """
        # 1. Spatial Distance
        dx = q1[0] - q0[0]
        dy = q1[1] - q0[1]
        d_space = np.sqrt(dx**2 + dy**2) # Euclidean distance in space

        # 2. Time Difference (Direction Dependent) (d_time should remain positive in either case)
        # read d_time as "difference in time"
        if forward:
            d_time = q1[2] - q0[2] 
        else:
            d_time = q0[2] - q1[2] # going backward in time

        # 3. Time Validation (Cannot go backward in time)
        # This should not happen due to reachable() check, but just in case
        if d_time <= 0:
            return q0

        # 4. Calculate Required Speed to hit the point exactly
        v_req = d_space / d_time

        # 5. Determine Step Size and Speed
        # We fly at V_MAX if:
        #   A. We physically can't reach the target in time (v_req > vmax)
        #      (This should be prevented by reachable() check, but just in case)
        #   B. OR The target is further away in time than our step size (d_time > max_time)
        if v_req > self.vmax or d_time > max_time:
            # Take a full step (or as much time as available)
            dt_step = min(d_time, max_time)
            # Fly as fast as possible to cover maximum distance
            dist_step = self.vmax * dt_step
            
        else:
            # Case C: The point is Reachable (v_req <= vmax) AND Close (d_time <= max_time).
            # We can reach the target exactly in this step.
            dt_step = d_time
            dist_step = d_space # Effectively v_req * dt_step

        # 6. Calculate New Position

        # checking that time step is positive
        # This shouldn't trigger due to previous checks, but just in case
        if dt_step <= 0: 
            return q0
        
        # Interpolate Space
        if d_space > 0:
            xnew = q0[0] + (dx / d_space) * dist_step
            ynew = q0[1] + (dy / d_space) * dist_step
        else:
            # Hovering (Spatial distance is 0, but we consumed time)
            xnew = q0[0]
            ynew = q0[1]

        # Interpolate Time
        if forward:
            tnew = q0[2] + dt_step
        else:
            tnew = q0[2] - dt_step

        return [xnew, ynew, tnew]


    # # Original extedn function. Has issue with keeping consistent units
    # def extend(self, q0, q1, max_time): # Original version
    #     """
    #     Extend edge from q0 to q1
    #     """
    #     dx = q1[0]-q0[0]
    #     dy = q1[1]-q0[1]
    #     dt = q1[2]-q0[2]
    #     max_distance = max_time*self.vmax

    #     if np.abs(dx)<=max_distance and \
    #     np.abs(dy)<=max_distance and \
    #     np.abs(dt)<=max_time:
    #         qnew = q1
    #     else:
    #         norm = np.sqrt(dx**2+dy**2+dt**2)
    #         dxnorm = dx/norm*self.vmax
    #         dynorm = dy/norm*self.vmax
    #         dtnorm = dt/norm*self.vmax
    #         qnew = [q0[0]+dxnorm, q0[1]+dynorm, q0[2]+dtnorm]
    #     return qnew

    def validate(self, q, vehicle_radius=None):
        """
        Validate the point while doing collision check
        """
        # True:  No collision
        # False: Collision detected

        if vehicle_radius is None:
            vehicle_radius = self.vehicle['radius']

        tGiven = q[2]

        # Generate vehicle with circular collision radius
        vehicle_centre = Point(q[0], q[1])
        vehicle_collision_bound = vehicle_centre.buffer(vehicle_radius)

        checkInMap = []
        # Check for map
        # Check if x coordinate is OK
        if q[0] >= 0 and q[0] <= self.map_size[1]:
            # Check if y coordinate is OK
            if q[1] >= 0 and q[1] <= self.map_size[3]:
                checkInMap = True
            else:
                checkInMap = False
        else:
            checkInMap = False

        # NEW: Check for static obstacles
        checkStatic = True
        num_static_obs = self.map_in['st']['n']
        for i in range(num_static_obs):
            # map_in['st'][str(i)] is an (N_verts, 2) numpy array of vertices
            poly_vertices = self.map_in['st'][str(i)]
            
            # # Create a Path object for the polygon
            # # This uses the vertices to define the boundary of the obstacle
            # obstacle_path = Path(poly_vertices)

            # Since the notebook defines the `map_in['st'][str(i)]` as an array of vertices,
            # we need to ensure we can create a shapely Polygon from it here.
            # Assuming you can import `Polygon` from `shapely.geometry` (like in the notebook cell 2 for `Point`):
            from shapely.geometry import Polygon
            obstacle_polygon = Polygon(poly_vertices)


            # Check if the vehicle's collision boundary intersects the static obstacle
            if vehicle_collision_bound.intersects(obstacle_polygon):
                 checkStatic = False
                 break # Collision found, no need to check other obstacles
            
        checkDynamic = True
        # Check for dynamic
        for i in range(self.cam_dict['n']):
            cameras = self.dmap.gen_cam(i, tGiven)
            cam_i = cameras[str(i)]['FOV_Poly']
            if cam_i.intersects(vehicle_collision_bound):

                # --- NEW: Check if the vehicle is hiding in a building's shadow ---
                is_occluded = False
                shadow_lines_for_this_cam = self.all_camera_shadows[i]
                
                for building_lines in shadow_lines_for_this_cam:
                    in_shadow = True
                    
                    for (a, b, c_line) in building_lines:
                        # Check if the center of the vehicle is behind the shadow line
                        # (A value >= 0 means inside the shadow wedge)
                        perp_dist = a * q[0] + b * q[1] + c_line
                        
                        if perp_dist < 0:
                            in_shadow = False
                            break # It is exposed past this line
                            
                    if in_shadow:
                        is_occluded = True
                        break # The vehicle is completely occluded by this building
                
                # If we checked all buildings and it wasn't occluded by ANY of them, it is detected!
                if not is_occluded:
                    checkDynamic = False 
                    break
        
        # True: Collision-Free
        # False: Collision
        return bool(checkInMap) and bool(checkDynamic) and bool(checkStatic) 

    def check_route(self, q1, q2, nInterpolate, vehicle_radius=None):
        """
        Check collision of generated path
        """
        if vehicle_radius is None:
            vehicle_radius = self.vehicle['radius']

        # Generate n number of Interpolated points to check collision
        qdiff = [q2[0]-q1[0], q2[1]-q1[1], q2[2]-q1[2]]
        interpolated = np.linspace(0, 1, nInterpolate)
        qxyz = [qdiff[0]*interpolated+q1[0], qdiff[1]*interpolated+q1[1], qdiff[2]*interpolated+q1[2]]

        # True: Collision-Free
        # False: Collision
        q12i_check = False
        for jj in range(nInterpolate):
            q12i =[qxyz[0][jj], qxyz[1][jj], qxyz[2][jj]]
            q12i_check = self.validate(q12i, vehicle_radius=vehicle_radius)
            if q12i_check == False:
                return False
        return True

    def find_path(self, x0, qfin, E, forward=True, tf=None):
        """
        Find path
        """
        path = []
        path.append(qfin)
        currentNode = qfin
        iter = 0
        while 1:
            for ei in range(len(E)):
                e = E[ei][1]
                
                if currentNode[0]==e[0] and currentNode[1]==e[1] and currentNode[2]==e[2]:
                    currentNode = E[ei][0]
                    path.append(currentNode)
                    break
            iter += 1
            if forward:
                if currentNode[0]==x0[0] and currentNode[1]==x0[1] and currentNode[2]==x0[2]:
                    return path
            else:
                if currentNode[0]==self.xf[0] and currentNode[1]==self.xf[1] and currentNode[2]==tf:
                    return path
                
    def find_neighbors_proximity(self, prox, q, V):
        """
        Find all neighbors within predetermined proximity
        """
        output = []
        for v in V:
            if 0 <= q[0]-v[0] <= prox and \
                0 <= q[1]-v[1] <= prox:
                output.append(v)
        return output

    """Parallel RRT in 2D Space-Time"""
    def path_planning_main(self, x0, nPartition, compTimeLimit, tf0, tfn):
        RRTP_total_time = []
        RRTP_total_distance = []
        total_path_time = 0
        total_path_distance = 0
        path_RRTP = None

        k = 0
        # Vertex
        V_RRTCa = []
        V_RRTCa.append([x0[0], x0[1], x0[2]])

        # Time Stamp
        T_RRTCa = []
        T_RRTCa.append(0)

        # Edge
        E_RRTCa = []

        # This generates the random end times for each RRT Tree
        tfSelection = []
        for ii in range(nPartition):
            tfSelection.append(rn.uniform(tf0, tfn))

        V_RRTCb = {}
        T_RRTCb = {}
        E_RRTCb = {}
        for ii in range(nPartition):
            V_RRTCb[str(ii)] = []
            V_RRTCb[str(ii)].append([self.xf[0], self.xf[1], tfSelection[ii]])
            T_RRTCb[str(ii)] = []
            T_RRTCb[str(ii)].append(tfSelection[ii])
            E_RRTCb[str(ii)] = []

        checkIfPathExist = False
        start = time.time()
        while time.time()-start <= compTimeLimit:
            for ii in range(nPartition):
                if time.time()-start > (compTimeLimit):
                    break
                currtf = tfSelection[ii]
                while 1:
                    print('k: '+str(k))
                    if time.time()-start > (compTimeLimit):
                        break
                    qrand = self.random_sample_test2(x0, currtf, 2, k)
                    
                    if time.time()-start > (compTimeLimit):
                        break
                    if np.mod(k,2) == 0:
                        qclosest = self.find_neighbor(qrand, V_RRTCa, 2, k)
                    else:
                        qclosest = self.find_neighbor(qrand, V_RRTCb[str(ii)], 2, k)
                    
                    if qclosest is not None:
                        if time.time()-start > (compTimeLimit):
                            break
                        
                        # updated version for extend() that enforces vmax
                        if np.mod(k, 2) == 0:
                            qnew = self.extend(qclosest, qrand, self.max_time, forward=True)  # Start tree
                        else:
                            qnew = self.extend(qclosest, qrand, self.max_time, forward=False)  # Goal tree
                       
                        # qnew = self.extend(qclosest, qrand, self.max_time) # Original version of extend

                        # Validate
                        validationCheck = self.validate(qnew)
                        if validationCheck is True and self.check_route(qclosest, qrand, nInterpolate=100, vehicle_radius=self.vehicle['radius']) is True:
                            break

                if np.mod(k,2) == 0:
                    V_RRTCa.append(qnew)
                    E_RRTCa.append([qclosest, qnew])
                    T_RRTCa.append(qnew[2])
                else:
                    V_RRTCb[str(ii)].append(qnew)
                    E_RRTCb[str(ii)].append([qclosest, qnew])
                    T_RRTCb[str(ii)].append(qnew[2])
                if time.time()-start > (compTimeLimit):
                    break
                # Check to continue:
                checkIfPathExist = False
                
                if k > 1:
                    qmin = qclosest
                    cost_old = self.distance(qclosest, qnew)
                    if np.mod(k,2) == 0:
                        neighbor_vector = self.find_neighbors_proximity(self.proximity_space, qnew, V_RRTCa)
                    else:
                        neighbor_vector = self.find_neighbors_proximity(self.proximity_space, qnew, V_RRTCb[str(ii)])

                    # Rewiring Step
                    if len(neighbor_vector) > 0:
                        for v in neighbor_vector:
                            if cost_old + self.distance(v, qnew) < self.distance(qclosest, v) and self.check_route(v, qnew, nInterpolate=100):
                                cost_old = cost_old + self.distance(v, qnew)
                                qmin = v

                    # Copilot update: Remove previous edge to qnew
                        if np.mod(k,2) == 0:
                            E_RRTCa = [e for e in E_RRTCa if e[1] != qnew]
                            E_RRTCa.append([qmin, qnew])
                        else:
                            E_RRTCb[str(ii)] = [e for e in E_RRTCb[str(ii)] if e[1] != qnew]
                            E_RRTCb[str(ii)].append([qmin, qnew])

                        # Original version, not removing any edges in rewiring :(        
                        # if np.mod(k,2) == 0:
                        #     E_RRTCa.append([qnew, qmin])
                        # else:
                        #     E_RRTCb[str(ii)].append([qnew, qmin])
                if time.time()-start > (compTimeLimit):
                    break
                if np.mod(k,2) == 0:
                
                    for vj in V_RRTCb[str(ii)]:

                        dist_spatial = self.distance(V_RRTCa[-1], vj)
                        dist_time = np.abs(V_RRTCa[-1][2]-vj[2]) # Directionality doesn't matter here, it's checked in reachable()

                        if dist_spatial <= self.proximity_space and dist_time <= self.proximity_time:
                            if self.reachable(V_RRTCa[-1], vj, forward=True):
                                if self.check_route(V_RRTCa[-1], vj, nInterpolate=100):
                                    connectEdge = [V_RRTCa[-1], vj]
                                    checkIfPathExist = True
                                    break
                else:
                    for vj in V_RRTCa:
                        dist_spatial = self.distance(vj, V_RRTCb[str(ii)][-1])
                        dist_time = np.abs(vj[2]-V_RRTCb[str(ii)][-1][2]) # Directionality doesn't matter here, it's checked in reachable()

                        if dist_spatial <= self.proximity_space and dist_time <= self.proximity_time:
                            if self.reachable(V_RRTCb[str(ii)][-1], vj, forward=False):
                                if self.check_route(vj, V_RRTCb[str(ii)][-1], nInterpolate=100):
                                    connectEdge = [vj, V_RRTCb[str(ii)][-1]]
                                    checkIfPathExist = True
                                    break
                k += 1
                if time.time()-start > (compTimeLimit):
                    break
                if checkIfPathExist:
                    pathRRTPa = self.find_path(x0, connectEdge[0], E_RRTCa, forward=True)
                    pathRRTPb = self.find_path(x0, connectEdge[1], E_RRTCb[str(ii)], forward=False, tf=tfSelection[ii])

                    # Combine path into one
                    # path_RRTP = list(reversed(pathRRTPa))+pathRRTPb[1:-1] # Original, no end point :(
                    path_RRTP = list(reversed(pathRRTPa))+pathRRTPb[1:]

                    for fpv in range(len(path_RRTP)-1):
                        total_path_time+=(path_RRTP[fpv+1][2]-path_RRTP[fpv][2])
                        total_path_distance+=self.distance(path_RRTP[fpv+1], path_RRTP[fpv])
                    RRTP_total_time.append(total_path_time)
                    RRTP_total_distance.append(total_path_distance)
                    break
                if time.time()-start > (compTimeLimit):
                    break
            if checkIfPathExist:
                break
            if time.time()-start > (compTimeLimit):
                break
        end = time.time()
        
        if checkIfPathExist:
            return True, end-start, total_path_distance, total_path_time, 1, path_RRTP, V_RRTCa, V_RRTCb, E_RRTCa, E_RRTCb
        else:
            return False, end-start, 0, 0, 0, None, V_RRTCa, V_RRTCb, E_RRTCa, E_RRTCb
        




    #def extend(self, q0, q1, max_time, forward=True):
        """
        Extend edge from q0 to q1 while enforcing vmax constraint.
        Handles both forward (start tree) and backward (goal tree) growth.
        """
        dx = q1[0] - q0[0]
        dy = q1[1] - q0[1]
        dd = np.sqrt(dx**2 + dy**2)

        # Compute required time to travel at vmax
        dt_required = dd / self.vmax

        # Clamp to max_time if needed
        dt = min(dt_required, max_time)

        # Ensure valid movement
        if dt <= 0 or dd == 0:
            return q0  # No movement possible

        # Normalize direction
        dxnorm = (dx / dd) * self.vmax * dt
        dynorm = (dy / dd) * self.vmax * dt

        # Adjust time based on direction
        if forward:
            tnew = q0[2] + dt
        else:
            tnew = q0[2] - dt

        xnew = q0[0] + dxnorm
        ynew = q0[1] + dynorm

        return [xnew, ynew, tnew]

    # def extend(self, q0, q1, max_time):
        """
        Extend edge from q0 to q1 while enforcing vmax constraint.
        """
        dx = q1[0] - q0[0]
        dy = q1[1] - q0[1]
        dd = np.sqrt(dx**2 + dy**2)

        # Compute required time to travel at vmax
        dt_required = dd / self.vmax

        # Clamp to max_time if needed
        dt = min(dt_required, max_time)

        # Ensure direction is correct
        if dt <= 0:
            return q0  # No movement possible

        # Compute new point
        xnew = q0[0] + (dx / dd) * self.vmax * dt
        ynew = q0[1] + (dy / dd) * self.vmax * dt
        tnew = q0[2] + dt

        return [xnew, ynew, tnew]