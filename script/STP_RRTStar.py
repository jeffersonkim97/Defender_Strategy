'''
STP-RRT algorithm with slight edits:
- fixed looping issue in rewiring step (now deleting old edges in rewiring)
- fixed issue with 'path' not including the goal point


Original code written by Jaehyeok Kim. Explained in his Masters Thesis

I'm just editing it into a stand-alone python file so I 
can use STP-RRT* in other files more cleanly :0
'''

# Required Python Packages
import numpy as np
import matplotlib.pylab as plt
import random as rn
import time
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from Dynamic import DynamicMap
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial import KDTree


class STP_RRTStar():

    def __init__(self, vmax, xf, map_size, vehicle, cam_dict, dmap, max_time, map_in):
        self.vmax = vmax
        self.xf = xf
        self.map_size = map_size
        self.vehicle = vehicle 
        self.cam_dict = cam_dict # dictionary of cameras
        self.dmap = dmap # dynamic map object
        self.max_time = max_time
        self.map_in = map_in
        print('STP-RRT* initialized')


    # ------------------------- STP-RRT* Subfunctions ----------------------------------------------#
    def distance(self, a, b):
        """
        Euclidean distance between two points
        """
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def random_sample(self, x0, tf, nn, k):
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
            yrand = rn.uniform(np.min([0, x0[0]/self.vmax]), np.min([self.vmax*(trand+x0[0]/self.vmax), self.map_size[1]]))

        return [xrand, yrand, trand]

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
        


    def extend(self, q0, q1, max_time): # Original version
        """
        Extend edge from q0 to q1
        """
        dx = q1[0]-q0[0]
        dy = q1[1]-q0[1]
        dt = q1[2]-q0[2]
        max_distance = max_time*self.vmax

        if np.abs(dx)<=max_distance and \
        np.abs(dy)<=max_distance and \
        np.abs(dt)<=max_time:
            qnew = q1
        else:
            norm = np.sqrt(dx**2+dy**2+dt**2)
            dxnorm = dx/norm*self.vmax
            dynorm = dy/norm*self.vmax
            dtnorm = dt/norm*self.vmax
            qnew = [q0[0]+dxnorm, q0[1]+dynorm, q0[2]+dtnorm]
        return qnew
    
    
    # Update: Extract angle condition of sensor i at given t
    def _bounce_angle(self, theta0, bound_plus, bound_minus, panspeed, t):
        """
        Bounce-pan angle inside [theta0+bound_minus, theta0+bound_plus] at time t.
        Angles are radians; panspeed is rad per unit time.
        """
        lo = theta0 + float(bound_minus)
        hi = theta0 + float(bound_plus)
        if lo > hi:
            lo, hi = hi, lo
        span = hi - lo
        if span <= 1e-12 or abs(panspeed) <= 1e-12:
            return theta0
        raw = theta0 + panspeed * t
        period = 2.0 * span
        off = (raw - lo) % period
        return (lo + off) if (off <= span) else (hi - (off - span))
    
    def detection_Cost(self, qa, qs, dt1, dt2, param_lambda, param_beta):
        """
        Compute detection cost: Elfes's model
        Inputs:
            qa: [x, y, t] attacker position
            qs: [x, y] sensor position
        """

        dist_def_to_atk = self.distance(qa, qs)

        if dist_def_to_atk <= dt1:
            detection_cost = 1
        elif dist_def_to_atk > dt1 and dist_def_to_atk < dt2:
            detection_cost = np.exp(-param_lambda*(dist_def_to_atk-dt1)**param_beta)
        else:
            detection_cost = 0

        return detection_cost
        

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
            
        if not checkInMap:
            return False, 0.0

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
                 return False, 0.0
        
        # Updated with detection cost for sensor FOVs    
        # Initialize detection cost
        cumulative_detection_cost = 0
        
        # Check for omni-directional sensor at fixed position
        n_omni = self.cam_dict['n_omni']
        if n_omni > 0:
            omni_sensor = self.cam_dict['omnidirectional']
            omni_sensor_detection_param = self.cam_dict['detection']['omnidirectional']
            for omni_i in range(n_omni):
                cumulative_detection_cost += self.detection_Cost(qa=q,
                                                                qs=[omni_sensor['x'][omni_i], omni_sensor['y'][omni_i]],
                                                                dt1=omni_sensor['spec']['fov'][1]/100,
                                                                dt2=omni_sensor['spec']['fov'][1],
                                                                param_lambda=omni_sensor_detection_param['param_lambda'],
                                                                param_beta=omni_sensor_detection_param['param_beta'])
        
        # Check for dynamic
        n_direc = self.cam_dict['n_direc']
        if n_direc > 0:
            direc_sensor = self.cam_dict['directional']
            direc_sensor_detection_param = self.cam_dict['detection']['directional']
            
            cx = np.array(direc_sensor['x'], dtype=float)
            cy = np.array(direc_sensor['y'], dtype=float)
            spec = direc_sensor['spec']
            init_angles = np.array(spec["init_angle"], dtype=float)
            bound_arr = np.array(spec["bound"], dtype=float)
            fov_half = float(spec["fov"][0])
            fov_range = float(spec["fov"][1])
            panspeeds = np.array(spec["panspeed"], dtype=float)
            
            for i in range(n_direc):
                def fov_sector_polygon(cx, cy, theta_rad, fov_half_rad, rng, n_arc=64):
                    # theta_rad: CCW from +x, radians
                    c, s = np.cos(theta_rad), np.sin(theta_rad)
                    R = np.array([[c, -s],[s,  c]])        # det +1 rotation
                    phis = np.linspace(-fov_half_rad, +fov_half_rad, int(n_arc))
                    arc_local = np.stack([rng*np.cos(phis), rng*np.sin(phis)], axis=1)  # (n,2)
                    arc_world = (R @ arc_local.T).T + np.array([cx, cy])
                    pts = np.vstack([[cx, cy], arc_world, [cx, cy]])  # apex → arc → apex
                    return Polygon(pts)

                theta = self._bounce_angle(init_angles[i], bound_arr[i,0], bound_arr[i,1], panspeeds[i], tGiven)  # radians, CCW
                cam_poly = fov_sector_polygon(cx[i], cy[i], theta, fov_half, fov_range)
                if cam_poly.covers(vehicle_collision_bound):
                    # checkDynamic = False
                    # break
                    cumulative_detection_cost += self.detection_Cost(qa=q, qs=[cx[i], cy[i]],
                                                                    dt1=direc_sensor['spec']['fov'][1]/100,
                                                                    dt2=direc_sensor['spec']['fov'][1],
                                                                    param_lambda=direc_sensor_detection_param['param_lambda'],
                                                                    param_beta=direc_sensor_detection_param['param_beta'])
        
        # True: Collision-Free
        # False: Collision
        return (bool(checkInMap) and bool(checkStatic)), cumulative_detection_cost

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
    
    ##########################################################################################
    ############################### Main STP-RRT* Function ###################################
    ##########################################################################################

    """Parallel RRT in 2D Space-Time"""
    def standard_Parallel_RRT(self, x0, nPartition, compTimeLimit, tf0, tfn, debug=False):
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
        qnew = x0
        while time.time()-start <= compTimeLimit:
            for ii in range(nPartition):
                if time.time()-start > (compTimeLimit):
                    break
                currtf = tfSelection[ii]
                while 1:
                    if debug:
                        if np.mod(k,2) == 0:
                            print('k: '+str(k)+', Convergence: '+str(self.distance(qnew, self.xf)))
                        else:
                            print('k: '+str(k)+', Convergence: '+str(self.distance(qnew, x0)))
                    if time.time()-start > (compTimeLimit):
                        break
                    qrand = self.random_sample(x0, currtf, 2, k)
                    
                    if time.time()-start > (compTimeLimit):
                        break
                    if np.mod(k,2) == 0:
                        qclosest = self.find_neighbor(qrand, V_RRTCa, 2, k)
                    else:
                        qclosest = self.find_neighbor(qrand, V_RRTCb[str(ii)], 2, k)
                    
                    if qclosest is not None:
                        if time.time()-start > (compTimeLimit):
                            break

                        qnew = self.extend(qclosest, qrand, self.max_time) # Original version of extend

                        # Validate
                        validationCheck, cost = self.validate(qnew)
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
                proximity = 0.1 #CHANGE THIS BASED ON MAP SIZE
                
                if k > 1:
                    # Heuristics for weight selection
                    detection_cost_weight = 2
                    distance_cost_weight = 1
                    
                    qmin = qclosest
                    temp, qclosest_cost = self.validate(qnew)
                    detection_cost_old = qclosest_cost
                    distance_cost_old = self.distance(qclosest, qnew)
                    if np.mod(k,2) == 0:
                        neighbor_vector = self.find_neighbors_proximity(proximity, qnew, V_RRTCa)
                    else:
                        neighbor_vector = self.find_neighbors_proximity(proximity, qnew, V_RRTCb[str(ii)])

                    # Rewiring Step
                    if len(neighbor_vector) > 0:
                        for v in neighbor_vector:
                            temp, v_cost = self.validate(v)
                            temp, qnew_cost = self.validate(qnew)
                            
                            # Cost computed as detection cost + distance cost
                            prev_detection_cost = (detection_cost_old+(qnew_cost-v_cost))*detection_cost_weight
                            prev_distance_cost = (distance_cost_old+self.distance(v,qnew))*distance_cost_weight
                            new_detection_cost = (v_cost-qclosest_cost)*detection_cost_weight
                            new_distance_cost = self.distance(qclosest,v)*distance_cost_weight
                            
                            if prev_detection_cost+prev_distance_cost < new_detection_cost+new_distance_cost and self.check_route(v, qnew, nInterpolate=100):
                                prev_detection_cost = detection_cost_old+(qnew-v_cost)
                                prev_distance_cost = distance_cost_old + self.distance(v,qnew)
                                qmin = v

                        # Copilot update: Remove previous edge to qnew
                            if np.mod(k,2) == 0:
                                E_RRTCa = [e for e in E_RRTCa if e[1] != qnew]
                                E_RRTCa.append([qmin, qnew])
                            else:
                                E_RRTCb[str(ii)] = [e for e in E_RRTCb[str(ii)] if e[1] != qnew]
                                E_RRTCb[str(ii)].append([qmin, qnew])

                if time.time()-start > (compTimeLimit):
                    break
                if np.mod(k,2) == 0:
                    for vj in V_RRTCb[str(ii)]:
                        if self.distance(V_RRTCa[-1], vj) <= proximity:
                            if self.reachable(V_RRTCa[-1], vj, forward=True):
                                if self.check_route(V_RRTCa[-1], vj, nInterpolate=100):
                                    connectEdge = [V_RRTCa[-1], vj]
                                    checkIfPathExist = True
                                    break
                else:
                    for vj in V_RRTCa:
                        if self.distance(vj, V_RRTCb[str(ii)][-1]) <= proximity:
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
        
    def detection_cost_for_path(self, path):
        total_detection_cost = 0
        for i in range(len(path)):
            temp, temp_cost = self.validate(path[i])
            total_detection_cost += temp_cost
        return total_detection_cost