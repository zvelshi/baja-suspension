# default
from __future__ import annotations
from typing import Dict

# third-party
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class DoubleAArmNumeric:
    def __init__(self, hp):
        self.hp = hp
        self.len = type(hp).link_lengths(hp)

        self._x_prev = np.hstack([hp.lbj, np.zeros(3)]) # seed guess
        self._wc0 = hp.wc[2]
        self._shock0 = self.len["shock_static"]
        self._tierod0 = self.len["tie_rod"]

        self.s_rel_pt = hp.ubj
        if hp.s_loc == 'upper':
            self.s_rel_pt = hp.ubj
            print("Assuming Upper A Arm mounted shock point...")
        else:
            self.s_rel_pt = hp.lbj
            print("Assuming Lower A Arm mounted shock point...")

    def reset(self):
        # reset the prev x to the default guess
        self._x_prev = np.hstack([self.hp.lbj, np.zeros(3)])

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
        if target_shock and not (hp.shock_min <= target_shock <= hp.shock_max):
            return None
    
        target_wheel = self._wc0 + bump_z if bump_z is not None else None

        target_tie   = self.len["tie_rod"]
        tr_ib_offset = hp.tr_ib + np.array([0.0, steer_mm, 0.0])

        # local coords (lbj frame)
        ubj_loc = hp.ubj - hp.lbj
        tr_ob_loc = hp.tr_ob - hp.lbj
        wc_loc = hp.wc - hp.lbj

        # local coords
        s_rel_pt = hp.ubj if hp.s_loc == 'upper' else hp.lbj
        sh_vec = hp.s_ob - s_rel_pt

        def res(x):
            p, e = x[:3], x[3:]
            Rw = self._rot(e)
            world = lambda v: p + Rw @ v

            lbj   = p
            ubj   = world(ubj_loc)
            tr_ob = world(tr_ob_loc)
            wc    = world(wc_loc)

            s_rel_pt_loc = ubj if hp.s_loc == 'upper' else lbj
            sha   = s_rel_pt_loc + sh_vec

            r = np.empty(6)
            
            # 4 a arms
            r[0] = np.linalg.norm(hp.uf - ubj) - self.len["upper_front"]
            r[1] = np.linalg.norm(hp.ur - ubj) - self.len["upper_rear"]
            r[2] = np.linalg.norm(hp.lf - lbj) - self.len["lower_front"]
            r[3] = np.linalg.norm(hp.lr - lbj) - self.len["lower_rear"]
            
            # tie-rod
            r[4] = np.linalg.norm(tr_ib_offset - tr_ob) - target_tie
            
            # shock / wheel
            if target_shock:
                r[5] = np.linalg.norm(hp.s_ib - sha) - target_shock
            else:
                r[5] = wc[2] - target_wheel

            return r

        lb = np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.pi/2])
        ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.pi/2])
        sol = least_squares(
            res, 
            self._x_prev, 
            bounds=(lb, ub), 
            xtol=1e-9,
        )
        if not sol.success:
            return
        self._x_prev = sol.x.copy()

        p, e = sol.x[:3], sol.x[3:]
        Rw   = self._rot(e) # wheel rot matrix
        world = lambda v: p + Rw @ v

        lbj = p
        ubj = world(ubj_loc)
        wc = world(wc_loc)
        tr_ob = world(tr_ob_loc)
        
        s_rel_pt_loc = ubj if hp.s_loc == 'upper' else lbj
        sha = s_rel_pt_loc + sh_vec

        step = {
            "lbj": lbj,
            "ubj": ubj,
            "wc": wc,
            "s_ob": sha,
            "tr_ib": tr_ib_offset,
            "tr_ob": tr_ob,
            "wheel_axis": Rw[:, 1],  # for plotting
        }
        
        # get wheel points
        step.update(self._rim_points(wc, Rw))

        # return resolved points
        return step