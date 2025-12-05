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
from scipy.spatial import KDTree


class STP_RRTStar():

    def __init__(self, vmax, xf, map_size, vehicle, cam_dict, dmap, max_time):
        self.vmax = vmax
        self.xf = xf
        self.map_size = map_size
        self.vehicle = vehicle 
        self.cam_dict = cam_dict # dictionary of cameras
        self.dmap = dmap # dynamic map object
        self.max_time = max_time
        print('8/27 5:23 PM version initialized')


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
            yrand = rn.uniform(np.min([0, x0[0]/self.vmax]), np.min([self.vmax*(trand+x0[0]/self.vmax), self.map_size[1]]))

        return [xrand, yrand, trand]

    def reachable(self, q0, q1, forward=True):
        """
        Check if q1 is within reachable set from q0
        """
        dx = np.abs(q1[0]-q0[0])
        dy = np.abs(q1[1]-q0[1])
        dd = np.sqrt(dx**2 + dy**2)
        dt = q1[2]-q0[2]

        if dd == 0:
            return True
        else:
            if forward and dt > 0 and dd/dt <= self.vmax:
                return True
            elif not forward and dt < 0 and dd/dt <= self.vmax:
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

    def extend(self, q0, q1, max_time): # Original version
        """
        Extend edge from q0 to q1
        """
        dx = q1[0]-q0[0]
        dy = q1[1]-q0[1]
        dt = q1[2]-q0[2]
        max_distance = max_time

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
        
        checkDynamic = True
        # Check for dynamic
        for i in range(self.cam_dict['n']):
            cameras = self.dmap.gen_cam(i, tGiven)
            cam_i = cameras[str(i)]['FOV_Poly']
            if cam_i.intersects(vehicle_collision_bound) is True:
                checkDynamic = False
        
        # True: Collision-Free
        # False: Collision
        return bool(checkInMap) and bool(checkDynamic) #and bool(checkStatic) 

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
    def standard_Parallel_RRT(self, x0, nPartition, compTimeLimit, tf0, tfn):
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
                        # if np.mod(k, 2) == 0:
                        #     qnew = self.extend(qclosest, qrand, self.max_time, forward=True)  # Start tree
                        # else:
                        #     qnew = self.extend(qclosest, qrand, self.max_time, forward=False)  # Goal tree
                       
                        qnew = self.extend(qclosest, qrand, self.max_time) # Original version of extend

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
                proximity = 0.1 #CHANGE THIS BASED ON MAP SIZE
                
                if k > 1:
                    qmin = qclosest
                    cost_old = self.distance(qclosest, qnew)
                    if np.mod(k,2) == 0:
                        neighbor_vector = self.find_neighbors_proximity(proximity, qnew, V_RRTCa)
                    else:
                        neighbor_vector = self.find_neighbors_proximity(proximity, qnew, V_RRTCb[str(ii)])

                    # Rewiring Step
                    if len(neighbor_vector) > 0:
                        for v in neighbor_vector:
                            if cost_old + self.distance(v, qnew) < self.distance(qclosest, v) and self.check_route(v, qnew):
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
        
