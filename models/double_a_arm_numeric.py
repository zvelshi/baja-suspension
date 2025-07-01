# default
from __future__ import annotations
from typing import Dict

# third-party
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ours
from scripts.hardpoints import DoubleAArmHardpoints

class DoubleAArmNumeric:
    """Numeric kinematics for a single double‑A‑arm corner."""

    def __init__(self, hp: DoubleAArmHardpoints):
        self.hp = hp
        self.len = DoubleAArmHardpoints.link_lengths(hp)
        self._x_prev = np.hstack([hp.bjl, np.zeros(3)]) # seed guess
        self._wc0 = hp.wc[2]
        self._shock0 = self.len["shock_static"]
        self._tierod0 = self.len["tie_rod"]

    def reset(self):
        # reset the prev x to the default guess
        self._x_prev = np.hstack([self.hp.bjl, np.zeros(3)])

    @staticmethod
    def _rot(eul: np.ndarray) -> np.ndarray:
        return R.from_euler("xyz", eul).as_matrix()
    
    def _rim_points(self, wc: np.ndarray, Rw: np.ndarray) -> Dict[str, np.ndarray]:
        # wheel axis in world
        n = Rw[:, 1]
        
        def project(axis: np.ndarray):
            v = axis - np.dot(axis, n) * n
            l2 = np.dot(v, v)
            return None if l2 < 1e-9 else wc + v / np.sqrt(l2) * self.hp.wr
        
        pts = {}
        xp = project(np.array([1., 0., 0.]))
        if xp is not None:
            pts["W_Xp"] = xp
            pts["W_Xm"] = wc - (xp - wc)

        # should never be needed
        yp = project(np.array([0., 1., 0.]))
        if yp is not None:
            pts["W_Yp"] = yp
            pts["W_Ym"] = wc - (yp - wc)

        zp = project(np.array([0., 0., 1.]))
        if zp is not None:
            pts["W_Zp"] = zp
            pts["W_Zm"] = wc - (zp - wc)

        return pts
    
    def solve(
            self,
            travel_mm : float | None = None,
            bump_z    : float | None = None,
            steer_mm  : float = 0.0,
        ):        
        if (travel_mm is None) == (bump_z is None):
                raise ValueError("Specify exactly ONE of travel_mm or bump_z")

        hp = self.hp
        target_shock = self._shock0 - travel_mm if travel_mm is not None else None
        target_wheel = self._wc0 + bump_z if bump_z is not None else None

        target_tie   = self.len["tie_rod"]
        tr_chassis_offset = hp.tr_chassis + np.array([0.0, steer_mm, 0.0])

        # local coords (bjl frame)
        bju_loc  = hp.bju - hp.bjl
        tr_up_loc= hp.tr_upright  - hp.bjl
        wc_loc   = hp.wc - hp.bjl

        # local coords (bju frame)
        sh_vec = hp.shock_a_arm - hp.bju

        def res(x):
            p, e = x[:3], x[3:]
            Rw   = self._rot(e)
            world = lambda v: p + Rw @ v

            bjl   = p
            bju   = world(bju_loc)
            tr_up = world(tr_up_loc)
            sha   = bju + sh_vec
            wc    = world(wc_loc)

            r = np.empty(6)
            
            # 4 a arms
            r[0] = np.linalg.norm(hp.uf - bju) - self.len["upper_front"]
            r[1] = np.linalg.norm(hp.ur - bju) - self.len["upper_rear"]
            r[2] = np.linalg.norm(hp.lf - bjl) - self.len["lower_front"]
            r[3] = np.linalg.norm(hp.lr - bjl) - self.len["lower_rear"]
            
            # tie-rod
            r[4] = np.linalg.norm(tr_chassis_offset - tr_up) - target_tie
            
            # shock / wheel
            r[5] = (np.linalg.norm(hp.shock_chassis - sha) - target_shock if travel_mm is not None else wc[2] - target_wheel)

            return r

        lb = np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.pi/2])
        ub = np.array([ np.inf, np.inf,  np.inf,  np.inf,  np.inf,  np.pi/2])
        sol = least_squares(res, self._x_prev, bounds=(lb, ub), xtol=1e-9)
        if not sol.success:
            return
            # raise RuntimeError(sol.message)
        self._x_prev = sol.x.copy()

        p, e = sol.x[:3], sol.x[3:]
        Rw   = self._rot(e) # wheel rot matrix
        world = lambda v: p + Rw @ v

        lbj   = p
        ubj   = world(bju_loc)
        wc    = world(wc_loc)
        sha   = ubj + sh_vec
        tr_up = world(tr_up_loc)

        shock_len = float(np.linalg.norm(hp.shock_chassis - sha))
        if not (hp.shock_min <= shock_len <= hp.shock_max):
            # print(f"shock length {shock_len:.1f} mm outside [{hp.shock_min},{hp.shock_max}] mm for travel={travel_mm:+.1f} mm, steer={steer_mm:+.1f} mm")
            return
        
        step = {
            "lower_ball_joint": lbj,
            "upper_ball_joint": ubj,
            "wheel_center"    : wc,
            "shock_a_arm"     : sha,
            "tie_rod_chassis" : tr_chassis_offset,
            "tie_rod_upright" : tr_up,
        }
        
        # get wheel points
        step.update(self._rim_points(wc, Rw))

        # return resolved points
        return step