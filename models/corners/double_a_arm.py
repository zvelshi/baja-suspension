# default
from __future__ import annotations

# ours
from models.joints.axle import Axle
from models.joints.cv_joint import CVJoint, PlungingCVJoint

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

        # passive axle
        self.axle_static_len = np.linalg.norm(hp.piv_ob - hp.piv_ib)
        self.piv_ob_loc = hp.piv_ob - hp.lbj

        self.axle = Axle(
            joint1=PlungingCVJoint(max_angle=30, plunge_limit=30.0), # Inboard slider
            joint2=CVJoint(max_angle=30), # Outboard fixed
            length=self.axle_static_len
        )

    def reset(self):
        # reset the prev x to the default guess
        self._x_prev = np.hstack([self.hp.lbj, np.zeros(3)])

    @staticmethod
    def _rot(eul: np.ndarray) -> np.ndarray:
        return R.from_euler("xyz", eul).as_matrix()
    
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

        # axle calcs
        piv_ob = world(self.piv_ob_loc)
        
        n_ib_dir = 1.0 if hp.piv_ib[1] > 0 else -1.0
        n_ib = np.array([0.0, n_ib_dir, 0.0])
        n_ob_dir = -1.0 if hp.wc[1] > 0 else 1.0
        n_ob = Rw @ np.array([0.0, n_ob_dir, 0.0])

        axle_state = self.axle.get_state(hp.piv_ib, piv_ob, n_ib, n_ob)

        step = {
            "lbj": lbj,
            "ubj": ubj,
            "wc": wc,
            "s_ob": sha,
            "tr_ib": tr_ib_offset,
            "tr_ob": tr_ob,
            "wheel_axis": Rw[:, 1],  # for plotting
            "axle_data": axle_state,
        }
        return step