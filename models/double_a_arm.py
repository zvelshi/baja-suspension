# default
from dataclasses import dataclass
from typing import Dict, Tuple

# third-party
import yaml
import numpy as np
from scipy.optimize import root
from scipy.spatial.transform import Rotation as R

@dataclass(frozen=True)
class DoubleAArmHardpoints:
    """All points are of type `np.ndarray` in chassis frame. hardpoints for left side."""

    # inboard a arm points
    uf: np.ndarray # upper front
    ur: np.ndarray # upper rear
    lf: np.ndarray # lower front
    lr: np.ndarray # lower rear

    # upright joints (at ride height)
    ubj: np.ndarray # upper ball joint
    lbj: np.ndarray # lower ball joint

    # steering points
    tr_chassis: np.ndarray # tie rod chassis
    tr_upright: np.ndarray # tie rod upright

    # shock points
    shock_chassis: np.ndarray # shock chassis
    shock_a_arm: np.ndarray # shock a arm

    # wheel points
    wc: np.ndarray # wheel center
    ws: np.ndarray # wheel spindle

    @classmethod
    def from_yml(self, yml_path: str) -> "DoubleAArmHardpoints":
        """Load hardpoints from a YAML file."""
        with open(yml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        return DoubleAArmHardpoints(
            uf=np.array(data['upper_front']),
            ur=np.array(data['upper_rear']),
            lf=np.array(data['lower_front']),
            lr=np.array(data['lower_rear']),
            ubj=np.array(data['upper_ball_joint']),
            lbj=np.array(data['lower_ball_joint']),
            tr_chassis=np.array(data['tie_rod_chassis']),
            tr_upright=np.array(data['tie_rod_upright']),
            shock_chassis=np.array(data['shock_chassis']),
            shock_a_arm=np.array(data['shock_a_arm']),
            wc=np.array(data['wheel_center']),
            ws=np.array(data['wheel_spindle']),
        )
    
    @classmethod
    def link_lengths(self, hp: "DoubleAArmHardpoints") -> Dict[str, float]:
        """Calculate link lengths based on hardpoints."""
        return {
            "link_upper_front": np.linalg.norm(hp.ubj - hp.uf),
            "link_upper_rear": np.linalg.norm(hp.ubj - hp.ur),
            "link_lower_front": np.linalg.norm(hp.lbj - hp.lf),
            "link_lower_rear": np.linalg.norm(hp.lbj - hp.lr),
            "link_tie_rod": np.linalg.norm(hp.tr_chassis - hp.tr_upright),
            "link_shock_static": np.linalg.norm(hp.shock_chassis - hp.shock_a_arm),
        }

class DoubleAArm:
    """
    Solve forward kinematics for a single corner.

    Args:
        hp (DoubleAArmHardpoints): Hardpoints for the corner. 
    """

    def __init__(self, hp: DoubleAArmHardpoints):
        self.hp = hp
        self.link_lengths = DoubleAArmHardpoints.link_lengths(hp)

        self.lbj_ref = hp.lbj.copy() # ride-height LBJ
        self._t_prev = self.lbj_ref
        self._r_prev = R.identity()

    @staticmethod
    def _apply(t: np.ndarray, r: R, p: np.ndarray, lbj_ref: np.ndarray):
        """
        Rigid-body transform p (chassis frame) with origin at lbj_ref.
        

        Args:
            t (np.ndarray): Translation vector (3D).
            r (R): Rotation as a scipy Rotation object.
            p (np.ndarray): Point in chassis frame to transform.
            lbj_ref (np.ndarray): Reference point for the lower ball joint.

        Returns:
            np.ndarray: Transformed point in the world frame.
        """
        return t + r.apply(p - lbj_ref)
    
    @staticmethod
    def pose_to_points(pose, hp, lbj_ref):
        t, r = pose
        A = lambda p: DoubleAArm._apply(t, r, p, lbj_ref)
        return {
            "ubj":         A(hp.ubj),
            "lbj":         A(hp.lbj),          # will be exactly t
            "tr_upright":  A(hp.tr_upright),
            "shock_a_arm": A(hp.shock_a_arm),
            "wc":          A(hp.wc),
            "ws":          A(hp.ws),
        }

    @staticmethod
    def error_func(
        vars_: np.ndarray,
        hp: DoubleAArmHardpoints,
        link_lengths: Dict[str, float],
        travel: float,
        lbj_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Non-linear error function for the double A-arm suspension.

        Args:
            vars_ (np.ndarray): [tx, ty, tz, rx, ry, rz]
            hp (DoubleAArmHardpoints): hardpoints for the corner.
            link_lengths (Dict[str, float]): Link lengths for the suspension.
            travel (float): Shock travel in meters (+) for compression, (-) for rebound.
            lbj_ref (np.ndarray): Reference point for the lower ball joint.
        """
    
        t = vars_[:3]
        r = R.from_rotvec(vars_[3:6])

        pts = DoubleAArm.pose_to_points((t, r), hp, lbj_ref)
        f = []

        # a arms
        f.append(np.linalg.norm(pts['ubj'] - hp.uf) - link_lengths['link_upper_front'])
        f.append(np.linalg.norm(pts['ubj'] - hp.ur) - link_lengths['link_upper_rear'])
        f.append(np.linalg.norm(pts['lbj'] - hp.lf) - link_lengths['link_lower_front'])
        f.append(np.linalg.norm(pts['lbj'] - hp.lr) - link_lengths['link_lower_rear'])

        # tie rod (fixed length)
        f.append(np.linalg.norm(pts['tr_upright'] - hp.tr_chassis) - link_lengths['link_tie_rod'])

        # wanted length = static_len - travel (positive travel = bump)
        f.append(np.linalg.norm(pts['shock_a_arm'] - hp.shock_chassis) - (link_lengths['link_shock_static'] + travel))

        return np.asarray(f)
    
    def solve(self, travel: float, steer: float) -> DoubleAArmHardpoints:
        """
        Solve forward kinematics for the double A-arm suspension.
        
        Args:
            travel (float): Shock travel in meters (+) for compression, (-) for rebound
            steer (float): Steer rack travel in meters (+) for left, (-) for right

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the positions of the hardpoints after applying travel and steering.
        """

        hp = self.hp

        # move rack inboard np.ndarray in +y
        hp_mod = hp.__class__(**{**hp.__dict__, "tr_chassis": hp.tr_chassis + np.array([0, steer, 0])})

        # initial guess
        x0 = np.hstack([self._t_prev, self._r_prev.as_rotvec()])

        sol = root(
            DoubleAArm.error_func,
            x0,
            args=(hp_mod, self.link_lengths, travel, self.lbj_ref),
            method="hybr",
            tol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"Root finding failed: {sol.message}")
        
        # unpack solution
        t_sol = sol.x[:3]
        r_sol = R.from_rotvec(sol.x[3:6])

        self._t_prev = t_sol
        self._r_prev = r_sol

        pts = DoubleAArm.pose_to_points((t_sol, r_sol), hp_mod, self.lbj_ref)

        return DoubleAArmHardpoints(
            uf=hp.uf, # hardpoints remain unchanged
            ur=hp.ur,
            lf=hp.lf,
            lr=hp.lr,
            ubj=pts['ubj'], # updated positions
            lbj=pts['lbj'],
            tr_chassis=hp_mod.tr_chassis, # modified tie rod chassis position
            tr_upright=pts['tr_upright'], # updated tie rod upright position
            shock_chassis=hp.shock_chassis, # unchanged shock chassis position
            shock_a_arm=pts['shock_a_arm'], # updated shock a arm position
            wc=pts['wc'], # updated wheel center position
            ws=pts['ws'], # updated wheel spindle position
        )