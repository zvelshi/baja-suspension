# default
from dataclasses import dataclass
from typing import Dict

# third-party
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
from scipy.spatial.transform import Rotation as R

@dataclass(frozen=True)
class DoubleAArmHardpoints:
    """All points are of type `np.ndarray` in chassis frame. hardpoints for left side."""

    # inboard a arm points
    uf: np.ndarray # upper front
    ur: np.ndarray # upper rear
    lf: np.ndarray # lower front
    lr: np.ndarray # lower rear

    # upright joints
    bju: np.ndarray # upper ball joint
    bjl: np.ndarray # lower ball joint

    # steering points
    tr_chassis: np.ndarray # tie rod chassis
    tr_upright: np.ndarray # tie rod upright

    # shock points
    shock_chassis: np.ndarray # shock chassis
    shock_a_arm: np.ndarray # shock a arm

    # wheel points
    wc: np.ndarray  # wheel center point
    ws: np.ndarray  # wheel spindle
    wr: float       # wheel radius
    ww: float       # wheel width

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
            bju=np.array(data['ball_joint_upper']),
            bjl=np.array(data['ball_joint_lower']),
            tr_chassis=np.array(data['tie_rod_chassis']),
            tr_upright=np.array(data['tie_rod_upright']),
            shock_chassis=np.array(data['shock_chassis']),
            shock_a_arm=np.array(data['shock_a_arm']),
            wc=np.array(data['wheel_center']),
            ws=np.array(data['wheel_spindle']),
            wr=data['wheel_radius'],
            ww=data['wheel_width'],
        )
    
    @classmethod
    def link_lengths(self, hp: "DoubleAArmHardpoints") -> Dict[str, float]:
        """Calculate link lengths based on hardpoints."""

        return {
            "link_upper_front": np.linalg.norm(hp.bju - hp.uf),
            "link_upper_rear": np.linalg.norm(hp.bju - hp.ur),
            "link_lower_front": np.linalg.norm(hp.bjl - hp.lf),
            "link_lower_rear": np.linalg.norm(hp.bjl - hp.lr),
            "link_tie_rod": np.linalg.norm(hp.tr_chassis - hp.tr_upright),
            "link_shock_static": np.linalg.norm(hp.shock_chassis - hp.shock_a_arm),
        }
    
    @staticmethod
    def points_yz(hp: "DoubleAArmHardpoints"):
        """Return DoubleAArmHardpoints dataclass with all points in yz plane. [y, z]"""

        return DoubleAArmHardpoints(
            uf=np.array([hp.uf[1], hp.uf[2]]),
            ur=np.array([hp.ur[1], hp.ur[2]]),
            lf=np.array([hp.lf[1], hp.lf[2]]),
            lr=np.array([hp.lr[1], hp.lr[2]]),
            bju=np.array([hp.bju[1], hp.bju[2]]),
            bjl=np.array([hp.bjl[1], hp.bjl[2]]),
            tr_chassis=np.array([hp.tr_chassis[1], hp.tr_chassis[2]]),
            tr_upright=np.array([hp.tr_upright[1], hp.tr_upright[2]]),
            shock_chassis=np.array([hp.shock_chassis[1], hp.shock_chassis[2]]),
            shock_a_arm=np.array([hp.shock_a_arm[1], hp.shock_a_arm[2]]),
            wc=np.array([hp.wc[1], hp.wc[2]]),
            ws=np.array([hp.ws[1], hp.ws[2]]),
            wr=np.array(hp.wr),
            ww=np.array(hp.ww),
        )

class DixonDoubleAArm:
    """Double A-Arm suspension kinematics solver using the Dixon method."""

    def __init__(self, hp: DoubleAArmHardpoints):
        """Initialize with hardpoints."""
        self.hp = hp
        self.link_lengths = DoubleAArmHardpoints.link_lengths(hp)

    def front_view_analysis(self, hp: DoubleAArmHardpoints, shock_bump: float) -> None:
        # drop x component of points, points are now [y, z]
        hp = DoubleAArmHardpoints.points_yz(hp)

        # get mid inboard a arm points
        lm = (hp.lf + hp.lr) / 2  # A - the midpoint of the lower arm inboard points
        um = (hp.uf + hp.ur) / 2  # C - the midpoint of the upper arm inboard points

        print("Lower arm midpoint (lm):", lm)
        print("Upper arm midpoint (um):", um)

        # static angles, unchanging through bump
        theta_ab0 = np.arctan2(hp.bjl[1] - lm[1], hp.bjl[0] - lm[0])
        theta_bd0 = np.arctan2((hp.bju[1] - hp.bjl[1]), (hp.bju[0] - hp.bjl[0]))

        print("Static angle of lower arm (theta_ab0):", theta_ab0)
        print("Static angle of ball joint axis (theta_bd0):", theta_bd0)

        l_ab = np.linalg.norm(lm - hp.bjl)      # length of lower arm link
        l_cd = np.linalg.norm(hp.bju - um)      # length of upper arm link
        l_bd = np.linalg.norm(hp.bju - hp.bjl)  # length of ball joint axis
        l_s0 = np.linalg.norm(hp.shock_chassis - hp.shock_a_arm)  # static shock length

        print("Lower arm length (l_ab):", l_ab)
        print("Upper arm length (l_cd):", l_cd)
        print("Ball joint axis length (l_bd):", l_bd)

        # rotate upper a arm until shock length = l_s0 - shock_bump
        target_l_s = l_s0 - shock_bump

        # create a 2x2 rotation matrix for angle phi
        R2 = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th),  np.cos(th)]])

        def shock_err(phi):
            """diff between desired and current shock length."""
            shock_a_arm = um + R2(phi) @ (hp.shock_a_arm - um)
            return np.linalg.norm(hp.shock_chassis - shock_a_arm) - target_l_s
        

        # initial guess for phi
        phi_L, phi_R = np.deg2rad(-35), np.deg2rad(35)  # lower and upper bounds for phi
        fL, fR = shock_err(phi_L), shock_err(phi_R)

        # 20-intersection bisection method to find the angle phi that satisfies the shock length condition
        for _ in range(20):
            phi_M = 0.5 * (phi_L + phi_R)
            fM = shock_err(phi_M)
            if np.abs(fM) < 1e-6:
                break
            if fL * fM < 0:
                phi_R, fR = phi_M, fM
            else:
                phi_L, fL = phi_M, fM
        phi = phi_M # the angle to rotate the upper arm to achieve the desired shock length
                
        # new locations of the arm shock eye and upper bj
        M = um + R2(phi) @ (hp.shock_a_arm - um)
        D = um + R2(phi) @ (hp.bju - um)

        v = D - lm
        d = np.hypot(v[0], v[1])  # distance from A to C
        a = (l_ab**2 - l_bd**2 + d**2) / (2 * d)  # distance from A to the intersection point along the line AC
        h = np.sqrt(max(l_ab**2 - a**2, 0.0)) # height of the intersection point above the line AC

        P2 = lm + a * v / d  # intersection point along the line AC
        perp = np.array([-v[1], v[0]]) / d  # perpendicular vector to AC
        B1, B2 = P2 + h * perp, P2 - h * perp  # two possible intersection points
        B = B1 if np.linalg.norm(B1 - hp.bjl) < np.linalg.norm(B2 - hp.bjl) else B2

        # wheel center point
        theta_bd = np.arctan2(D[1] - B[1], D[0] - B[0]) # new upright angle
        dtheta = theta_bd - theta_bd0
        d_WB = hp.wc - hp.bjl # vector from lower ball joint to wheel center
        bwc = B + R2(dtheta) @ d_WB # bumped wheel center

        # print results
        fig, ax = plt.subplots(figsize=(10, 10))

        # plot the points
        ax.scatter(lm[0], lm[1], color='black', label='Lower Arm Midpoint (A)', s=100)
        ax.scatter(um[0], um[1], color='black', label='Upper Arm Midpoint (C)', s=100)
        ax.scatter(hp.bjl[0], hp.bjl[1], color='green', label='Lower Ball Joint (b)', s=100)
        ax.scatter(hp.bju[0], hp.bju[1], color='green', label='Upper Ball Joint (d)', s=100)
        ax.scatter(B[0], B[1], color='blue', label='Lower Arm Outboard Point (B)', s=100)
        ax.scatter(D[0], D[1], color='red', label='Upper Arm Outboard Point (D)', s=100)
        ax.scatter(M[0], M[1], color='orange', label='Shock Outboard Point (M)', s=100)
        ax.scatter(hp.wc[0], hp.wc[1], color='purple', label='Static Wheel Center (WC)', s=100)
        ax.scatter(bwc[0], bwc[1], color='purple', label='Bumped Wheel Center (WC)', s=100)
   
        # plot the links
        ax.plot([hp.bjl[0], lm[0]], [hp.bjl[1], lm[1]], color='blue', label='Lower Arm (a-b)', linewidth=2)
        ax.plot([lm[0], B[0]], [lm[1], B[1]], color='blue', linestyle='--', label='Lower Arm Bump (a-b)', linewidth=2)
        ax.plot([hp.bju[0], um[0]], [hp.bju[1], um[1]], color='red', label='Upper Arm (c-d)', linewidth=2)
        ax.plot([um[0], D[0]], [um[1], D[1]], color='red', linestyle='--', label='Upper Arm Bump (c-d)', linewidth=2)
        ax.plot([B[0], D[0]], [B[1], D[1]], color='green', linestyle='--', label='New Ball Joint Axis (b-d)', linewidth=2)
        ax.plot([hp.bjl[0], hp.bju[0]], [hp.bjl[1], hp.bju[1]], color='green', label='Old Ball Joint Axis (b-d)', linewidth=2)
        ax.plot([hp.shock_chassis[0], M[0]], [hp.shock_chassis[1], M[1]], color='orange', linestyle='--', label='Shock (chassis-a arm)', linewidth=2)
        ax.plot([hp.shock_chassis[0], hp.shock_a_arm[0]], [hp.shock_chassis[1], hp.shock_a_arm[1]], color='orange', label='Shock (chassis-a arm static)', linewidth=2)
    
        ax.grid(True)
        ax.set_xlabel('Y Axis (mm)')
        ax.set_ylabel('Z Axis (mm)')
        ax.set_title(f'{-shock_bump}mm Shock Bump Analysis')
        ax.legend()
        ax.invert_xaxis()
        plt.show()