# default
from typing import Dict, Tuple

# third-party
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ours
from scripts.hardpoints import SemiTrailingLink

class SemiTrailingLinkNumeric:
    def __init__(self, hp: SemiTrailingLink):
        self.hp   = hp
        self.len  = SemiTrailingLink.link_lengths(hp)

        # state vector = [wc_x, wc_y, wc_z, eul_x, eul_y, eul_z]
        self._x_prev = np.hstack([hp.wc, np.zeros(3)])

        self._wc_z0   = hp.wc[2]
        self._shock_0 = self.len["shock_static"]

    def reset(self):
        self._x_prev = np.hstack([self.hp.wc, np.zeros(3)])

    @staticmethod
    def _rot(eul: np.ndarray) -> np.ndarray:
        return R.from_euler("xyz", eul).as_matrix()

    def _rim_points(self, wc: np.ndarray, Rw: np.ndarray) -> Dict[str, np.ndarray]:
        """Return up to ±X / ±Y / ±Z points on the rim for plotting."""
        n = Rw[:, 1]                    # wheel-rotation axis in world

        def project(axis: np.ndarray):
            v = axis - np.dot(axis, n) * n      # strip axial component
            l2 = np.dot(v, v)
            return None if l2 < 1e-9 else wc + v / np.sqrt(l2) * self.hp.wr

        pts: Dict[str, np.ndarray] = {}
        xp = project(np.array([1., 0., 0.]))
        if xp is not None:
            pts["W_Xp"] = xp
            pts["W_Xm"] = wc - (xp - wc)

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
        *,
        travel_mm: float | None = None,
        bump_z: float | None = None,
    ) -> Dict[str, np.ndarray]:

        if (travel_mm is None) == (bump_z is None):
            raise ValueError("Specify exactly *one* of travel_mm or bump_z")

        hp = self.hp
        targ_shock = self._shock_0 - travel_mm if travel_mm is not None else None
        if not (hp.shock_min <= targ_shock <= hp.shock_max):
            return None
        
        targ_wcz = self._wc_z0 + bump_z if bump_z is not None else None

        # Vectors from wheel center to outboard points in static position
        ucl_vec = hp.ucl_ob - hp.wc
        lcl_vec = hp.lcl_ob - hp.wc
        s_ob_vec = hp.s_ob - hp.wc
        
        # Vector from pivot inboard to outboard in static position
        piv_vec = hp.piv_ob - hp.piv_ib

        def res(x):
            p, e = x[:3], x[3:]
            Rw = self._rot(e)
            
            # Transform outboard points (rotate with wheel center)
            ucl_ob_w = p + Rw @ ucl_vec
            lcl_ob_w = p + Rw @ lcl_vec
            s_ob_w = p + Rw @ s_ob_vec
            
            # Pivot outboard rotates about inboard pivot
            piv_ob_w = hp.piv_ib + Rw @ piv_vec
            
            r = np.empty(6)
            # 1 - Lower trailing link length constraint
            r[0] = np.linalg.norm(hp.tl_f - ucl_ob_w) - self.len["upper_trailing_link"]

            # 2 - Upper trailing link length constraint
            r[1] = np.linalg.norm(hp.tl_f - lcl_ob_w) - self.len["lower_trailing_link"]
            
            # 3 - Upper camber link length
            r[2] = np.linalg.norm(ucl_ob_w - hp.ucl_ib) - self.len["upper_camber_link"]
            
            # 4 - Lower camber link length
            r[3] = np.linalg.norm(lcl_ob_w - hp.lcl_ib) - self.len["lower_camber_link"]
            
            # 5 - Wheel center to pivot outboard distance (rigid upright)
            r[4] = np.linalg.norm(p - piv_ob_w) - self.len["axle_ob_wc"]
                        
            # 6 - User constraint
            if targ_shock is not None:
                r[5] = np.linalg.norm(hp.s_ib - s_ob_w) - targ_shock
            else:
                r[5] = p[2] - targ_wcz
            
            return r

        sol = least_squares(
            res,
            self._x_prev,
            xtol=1e-9,
        )

        if not sol.success:
            print(f"Solution failed: {sol.message}")
            return None

        self._x_prev = sol.x.copy()
        p, e = sol.x[:3], sol.x[3:]
        Rw = self._rot(e)

        # Calculate all transformed points consistently
        ucl_ob_w = p + Rw @ ucl_vec
        lcl_ob_w = p + Rw @ lcl_vec
        s_ob_w = p + Rw @ s_ob_vec
        piv_ob_w = hp.piv_ib + Rw @ piv_vec  # Consistent with res() function

        # Return all points for visualization
        step = {
            'wc': p,
            'ucl_ib': hp.ucl_ib,
            'ucl_ob': ucl_ob_w,
            'lcl_ib': hp.lcl_ib,
            'lcl_ob': lcl_ob_w,
            'piv_ib': hp.piv_ib,
            'piv_ob': piv_ob_w,
            's_ib': hp.s_ib,
            's_ob': s_ob_w,
            'tl_f': hp.tl_f,
        }
        step.update(self._rim_points(p, Rw))
        return step