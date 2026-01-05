import numpy as np
from shapely import geometry
from shapely.geometry import Polygon
import casadi as ca

# Camera object class
# This is a class for single camera setup, which returns
# 1) Position of camera mounted, 2) FOV area at given time t
class Camera:
    def __init__(self, i, cam_dict, x0, y0):
        self.x0 = x0
        self.y0 = y0
        self.i = i
        cam_spec = cam_dict['directional']['spec']
        self.tilt_lim = cam_spec['bound'][i]
        self.fov_ang = cam_spec['fov'][0]
        self.Rc = cam_spec['fov'][1]
        self.init_angle = cam_spec['init_angle'][i]
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
    
    # def get_ctr_theta_t(self, t_in):
    #     # Compute angle of FOV centerline, bounded between two tilt limits
    #     # This is a continuous function, and returns centerline angle at that given t_in
    #     up = self.tilt_lim[0]
    #     down = self.tilt_lim[1]
    #     A = np.abs(up - down)
    #     h = A/2
    #     # B = 2*np.pi/self.cam_period
    #     B = 2*np.pi*self.cam_fovspeed

    #     return A/2*np.sin(B*t_in) + self.init_angle#+h
    # def get_ctr_theta_t(self, t_in):
    #     theta_min, theta_max = self.tilt_lim
    #     theta_mid = 0.5*(theta_min + theta_max)
    #     A = float(theta_max - theta_min)          # total span
    #     # Choose ONE of these depending on your units for panspeed:
    #     # (a) panspeed in Hz:
    #     B = 2*np.pi*self.cam_fovspeed
    #     # (b) panspeed in rad/s:
    #     # B = self.cam_fovspeed
    #     return theta_mid + 0.5*A*np.sin(B*t_in)
    # def get_ctr_theta_t(self, t_in):
    #     # self.tilt_lim is your two-element bound array (relative to init_angle)
    #     b1, b2 = float(self.tilt_lim[0]), float(self.tilt_lim[1])
    #     w = float(self.cam_fovspeed)  # rad/s (your JSON panspeeds look like rad/s)
    #     return self._bounce_angle(self.init_angle, b1, b2, w, float(t_in))

    def get_ctr_theta_t(self, t_in):
        b1 = self.tilt_lim[0]
        b2 = self.tilt_lim[1]
        w = self.cam_fovspeed
        
        # Calculate midpoint and amplitude
        mid = (b1 + b2) / 2
        amp = (b2 - b1) / 2
        
        # Smooth oscillation: angle = midpoint + amp * sin(frequency * t)
        # This is 100% CasADi compatible and differentiable
        return self.init_angle + mid + amp * ca.sin(w * t_in)

    
    # def get_fov(self, x0, y0, t_in):
    #     # Compute FOV at given time t_in
    #     th = self.get_ctr_theta_t(t_in)
    #     p1 = self.p1(x0,y0,self.fov_ang/2)
    #     p2 = self.p2(x0,y0,self.fov_ang/2)

    #     fov = np.vstack(([x0, y0], p1, p2))

    #     # Translate to origin
    #     trans = np.vstack([[x0, y0],[x0, y0],[x0, y0]])
    #     fov -= trans

    #     # Rotate to match boundaries
    #     R = np.array([
    #         [np.cos(th), -np.sin(th)],
    #         [np.sin(th), np.cos(th)]
    #     ])

    #     #print('rotation: ', np.rad2deg(th))
        
    #     # Rotate
    #     fov = fov@R

    #     # Translate to position
    #     fov += trans

    #     return fov

    # def gen_fov_polygon(self, x0, y0, t_in):
    #     fov = self.get_fov(x0, y0, t_in)

    #     fov_list = []
    #     for i in range(3):
    #         fov_list.append(fov[i,:])

    #     return geometry.Polygon(geometry.LineString(fov_list))


    def gen_fov_polygon(self, x_ctr: float, y_ctr: float, t: float):
        """
        Return a Shapely Polygon of the FOV wedge at time t.
        Uses get_ctr_theta_t(t), self.fov_half (rad), and self.Rc (range).
        """
        theta_c = self.get_ctr_theta_t(t)
        th1 = theta_c - self.fov_ang/2
        th2 = theta_c + self.fov_ang/2

        # 3-point wedge fan: center -> edge1 -> edge2 -> back to center
        p0 = (x_ctr, y_ctr)
        p1 = (x_ctr + self.Rc*np.cos(th1), y_ctr + self.Rc*np.sin(th1))
        p2 = (x_ctr + self.Rc*np.cos(th2), y_ctr + self.Rc*np.sin(th2))
        return Polygon([p0, p1, p2])

    # Backward-compatible alias: keep old name, same return type
    def get_fov(self, x_ctr: float, y_ctr: float, t: float):
        return self.gen_fov_polygon(x_ctr, y_ctr, t)
    
    # --- add this helper inside Camera.py (top-level or as @staticmethod) ---
    def _bounce_angle(self, theta0: float, b1: float, b2: float, w: float, t: float) -> float:
        """
        Reflect-sawtooth pan inside [theta0+min(b1,b2), theta0+max(b1,b2)].
        All angles in radians. w in rad/s.
        """
        lo = theta0 + min(b1, b2)
        hi = theta0 + max(b1, b2)
        span = hi - lo
        if span <= 1e-12 or abs(w) <= 1e-12:
            return float(np.clip(theta0, lo, hi))
        raw = theta0 + w*t
        period = 2.0*span
        off = (raw - lo) % period
        return (lo + off) if (off <= span) else (hi - (off - span))
