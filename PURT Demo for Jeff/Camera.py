# 2/10/26 Updated version using triangle wave for panning, 
# which is more realistic to actual cameras

import numpy as np
from shapely import geometry 
from scipy import signal

# Camera object class
# This is a class for single camera setup, which returns
# 1) Position of camera mounted, 2) FOV area at given time t
class Camera:
    def __init__(self, i, cam_dict, x0, y0):
        self.x0 = x0
        self.y0 = y0
        self.i = i
        cam_spec = cam_dict['spec']
        self.tilt_lim = cam_spec['bound'][i]
        self.fov_ang = cam_spec['fov'][0]
        self.Rc = cam_spec['fov'][1]
        self.init_angle = cam_dict['spec']['init_angle'][i]
        self.cam_period = cam_spec['cam_time'][0]
        self.cam_dt = cam_spec['cam_time'][1]
        self.cam_fovspeed = cam_spec['panspeed'][i]

    def cam_position(self):
        return (self.x0, self.y0)
    
    def p1(self, x0, y0, th):
        x1 = x0 + self.Rc*np.cos(th)
        y1 = y0 + self.Rc*np.sin(th)
        return [x1, y1]
    
    def p2(self, x0, y0, th):
        x2 = x0 + self.Rc*np.cos(th)
        y2 = y0 - self.Rc*np.sin(th)
        return [x2, y2]

    def get_ctr_theta_t(self, t_in):
        # ... existing setup code ...
        up = self.tilt_lim[0]
        down = self.tilt_lim[1]
        A = np.abs(up - down)
        B = 2 * np.pi * self.cam_fovspeed  # Ensure this matches your frequency/speed math

        # # width=0.5 creates a symmetrical triangle wave (equal time panning left vs right)
        # triangle_wave = signal.sawtooth(B * t_in + np.pi/2, width=0.5)

        # # ^^ This version breaks when using casadi MX variables, so here's a manual implementation:
        # triangle_wave = (2 / np.pi) * np.arcsin(np.sin(B * t_in - np.pi/2))

        # ^^ this version is not differentiable, so here is the smooth approximation using a Fourier series:
        # \frac{8}{\pi^{2}}\left(\sin\left(x\right)-\frac{1}{9}\sin\left(3x\right)+\frac{1}{25}\sin\left(5x\right)\right)
        triangle_wave = (8 / (np.pi**2)) * (
            np.sin(B * t_in) - (1/9)*np.sin(3*B * t_in) + (1/25)*np.sin(5*B * t_in)
        )
    
        return (A / 2) * triangle_wave + self.init_angle


    # def get_ctr_theta_t(self, t_in):
    #     # This version does sinusoidal panning, which is not realistic to the actual camera
    #     # I'm making a new version to do triangle-wave panning

    #     # Compute angle of FOV centerline, bounded between two tilt limits
    #     # This is a continuous function, and returns centerline angle at that given t_in
    #     up = self.tilt_lim[0]
    #     down = self.tilt_lim[1]
    #     A = np.abs(up - down)
    #     h = A/2
    #     # B = 2*np.pi/self.cam_period
    #     B = 2*np.pi*self.cam_fovspeed

    #     return A/2*np.sin(B*t_in) + self.init_angle#+h
    #     # # Updated version to realign fov triangles and bright spots of heat map in animation
    #     # return -A/2*np.sin(B*t_in) + self.init_angle#+h
    
    def get_fov(self, x0, y0, t_in):
        # Compute FOV at given time t_in
        th = self.get_ctr_theta_t(t_in)
        p1 = self.p1(x0,y0,self.fov_ang/2)
        p2 = self.p2(x0,y0,self.fov_ang/2)

        fov = np.vstack(([x0, y0], p1, p2))

        # Translate to origin
        trans = np.vstack([[x0, y0],[x0, y0],[x0, y0]])
        fov -= trans

        # Rotate to match boundaries
        # CRITICAL FIX: using -th to reverse the direction of rotation so that the visualization
        # lines up with the math
        R = np.array([
            [np.cos(-th), -np.sin(-th)],
            [np.sin(-th), np.cos(-th)] #Flip to -th again
        ])

        #print('rotation: ', np.rad2deg(th))
        
        # Rotate
        fov = fov@R

        # Translate to position
        fov += trans

        return fov

    def gen_fov_polygon(self, x0, y0, t_in):
        fov = self.get_fov(x0, y0, t_in)

        fov_list = []
        for i in range(3):
            fov_list.append(fov[i,:])

        return geometry.Polygon(geometry.LineString(fov_list))