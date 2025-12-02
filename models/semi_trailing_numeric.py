# default
from typing import Dict

# third-party
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ours
from models.axle import Axle, CVJoint, PlungingCVJoint

class SemiTrailingLinkNumeric:
    def __init__(self, hp):
        self.hp = hp
        self.len = type(hp).link_lengths(hp)

        # state vector = [wc_x, wc_y, wc_z, eul_x, eul_y, eul_z]
        self._x_prev = np.hstack([hp.wc, np.zeros(3)])

        self._wc_z0   = hp.wc[2]
        self._shock_0 = self.len["shock_static"]

        # ----------------- Axle setup -----------------

        axle_length = self.len["axle_ib_ob_static"]  # piv_ib <-> piv_ob
        axle_ob_wc  = self.len["axle_ob_wc"]         # piv_ob <-> wc

        # Choose inboard direction along +/- Y so "inboard" is towards y = 0
        y_ib = hp.piv_ib[1]
        if y_ib > 0:
            inboard_dir = -1.0
        elif y_ib < 0:
            inboard_dir = 1.0
        else:
            inboard_dir = 1.0  # arbitrary if exactly on centreline

        # Joint 1 extremity: 20 mm inboard of pivot_inboard along +/- y
        self._joint1_ext_offset = np.array([0.0, inboard_dir * 20.0, 0.0])

        # Plunging CV at the inboard end (joint 1)
        joint1 = PlungingCVJoint(
            ext_length=np.linalg.norm(self._joint1_ext_offset), # ≈ 20 mm
            max_angle=None,
            plunge_axis="y",
        )

        # Regular CV at the outboard end (joint 2)
        joint2 = CVJoint(
            ext_length=axle_ob_wc,
            max_angle=None,
        )

        self.axle = Axle(
            joint1=joint1,
            joint2=joint2,
            length=axle_length,
        )

        # Static vector from wheel centre to pivot outboard
        # (wheel centre is joint2 extremity, pivot outboard is joint2 centre)
        self._axle_ob_vec = hp.piv_ob - hp.wc

    def reset(self):
        self._x_prev = np.hstack([self.hp.wc, np.zeros(3)])

    @staticmethod
    def _rot(eul: np.ndarray) -> np.ndarray:
        return R.from_euler("xyz", eul).as_matrix()

    def _rim_points(self, wc: np.ndarray, Rw: np.ndarray) -> Dict[str, np.ndarray]:
        """Return up to ±X / ±Y / ±Z points on the rim for plotting."""
        n = Rw[:, 1]  # wheel-rotation axis in world

        def project(axis: np.ndarray):
            v = axis - np.dot(axis, n) * n  # strip axial component
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
    ) -> Dict[str, np.ndarray] | None:

        if (travel_mm is None) == (bump_z is None):
            raise ValueError("Specify exactly *one* of travel_mm or bump_z")

        hp = self.hp
        targ_shock = self._shock_0 - travel_mm if travel_mm is not None else None
        if targ_shock is not None and not (hp.shock_min <= targ_shock <= hp.shock_max):
            return None

        targ_wcz = self._wc_z0 + bump_z if bump_z is not None else None

        # Vectors from wheel center to outboard points in static position
        ucl_vec = hp.ucl_ob - hp.wc
        lcl_vec = hp.lcl_ob - hp.wc
        s_ob_vec = hp.s_ob - hp.wc

        # (wheel-centre -> pivot outboard) in static
        axle_ob_vec = self._axle_ob_vec

        joint1_ext_offset = self._joint1_ext_offset

        def res(x):
            p, e = x[:3], x[3:]
            Rw = self._rot(e)

            # Transform outboard points (rotate with wheel centre)
            ucl_ob_w = p + Rw @ ucl_vec
            lcl_ob_w = p + Rw @ lcl_vec
            s_ob_w   = p + Rw @ s_ob_vec

            # Pivot outboard moves with the wheel (rigid upright)
            piv_ob_w = p + Rw @ axle_ob_vec

            # Pivot inboard - fixed in world rotation at the inboard chassis point
            piv_ib_w = hp.piv_ib

            # Axle joint extremities
            joint1_ext_w = piv_ib_w + joint1_ext_offset
            joint2_ext_w = p 

            axle_res = self.axle.constraints(
                joint1_pos=piv_ib_w,
                joint1_ext=joint1_ext_w,
                joint2_pos=piv_ob_w,
                joint2_ext=joint2_ext_w,
            )

            length_err = axle_res["length"]
            j1_err     = axle_res["joint1"]
            j2_err     = axle_res["joint2"]

            r = np.empty(6)

            # 1 - Upper trailing link length
            r[0] = np.linalg.norm(hp.tl_f - ucl_ob_w) - self.len["upper_trailing_link"]

            # 2 - Lower trailing link length
            r[1] = np.linalg.norm(hp.tl_f - lcl_ob_w) - self.len["lower_trailing_link"]

            # 3 - Upper camber link length
            r[2] = np.linalg.norm(ucl_ob_w - hp.ucl_ib) - self.len["upper_camber_link"]

            # 4 - Lower camber link length
            r[3] = np.linalg.norm(lcl_ob_w - hp.lcl_ib) - self.len["lower_camber_link"]

            # 5 - Axle constraint
            r[4] = length_err + j1_err + j2_err

            # 6 - User constraint (shock length OR wheel-centre bump height)
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

        # Calculate all transformed points consistently for plotting
        ucl_ob_w = p + Rw @ ucl_vec
        lcl_ob_w = p + Rw @ lcl_vec
        s_ob_w   = p + Rw @ s_ob_vec
        piv_ob_w = p + Rw @ axle_ob_vec
        piv_ib_w = hp.piv_ib

        step = {
            "wc": p,
            "ucl_ib": hp.ucl_ib,
            "ucl_ob": ucl_ob_w,
            "lcl_ib": hp.lcl_ib,
            "lcl_ob": lcl_ob_w,
            "piv_ib": piv_ib_w,
            "piv_ob": piv_ob_w,
            "s_ib": hp.s_ib,
            "s_ob": s_ob_w,
            "tl_f": hp.tl_f,
            "wheel_axis": Rw[:, 1],  # for plotting
        }
        step.update(self._rim_points(p, Rw))
        return step