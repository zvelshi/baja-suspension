from __future__ import annotations

# default
import yaml
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ours
from models.corners.double_a_arm import DoubleAArmNumeric
from models.corners.semi_trailing_link import SemiTrailingLinkNumeric

# third-party
import numpy as np

class Vehicle:
    nickname: str

    def __init__(self, data: Dict = {}):
        self.nickname = list(data.keys())[0]
        vehicle_data = data[self.nickname]

        # (left/right, front/rear)
        self.front_left  = Corner(vehicle_data, (0, 0))
        self.front_right = Corner(vehicle_data, (1, 0))
        self.rear_left   = Corner(vehicle_data, (0, 1))
        self.rear_right  = Corner(vehicle_data, (1, 1))

    def run_simulation(self, simulation_class, **kwargs):
        simulation = simulation_class(self, kwargs.get("config", {}))
        return simulation.run(**kwargs)   # fix: return steps so caller can use them

class Corner:
    """
         (0, 0) _________ (1, 0)
                |       |
                |       |
                |       |
                |       |
                |       |
         (0, 1) |_______| (1, 1)
    """
    def __init__(self, data: Dict, pos: Tuple[int, int]):
        self.pos = pos  # (left/right, front/rear)

        if self.pos[1] == 0:
            hp = DoubleAArm.from_data(data=data['front'])
        else:
            hp = SemiTrailingLink.from_data(data=data['rear'])

        if self.pos[0] == 0:  # left side -> mirror across y-axis
            hp = type(hp).mirror_points(hp)

        hp._fill_vehicle_properties(data=data)

        self.hardpoints = hp
        self.solver = DoubleAArmNumeric(hp) if isinstance(hp, DoubleAArm) else SemiTrailingLinkNumeric(hp)

@dataclass
class Hardpoints:
    # shock properties
    shock_min: float = field(default=0.0, init=False)
    shock_max: float = field(default=0.0, init=False)

    # wheel properties
    wr: float        = field(default=0.0, init=False)
    ww: float        = field(default=0.0, init=False)

    def _fill_vehicle_properties(self, data):
        self.shock_min = data['shock_min']
        self.shock_max = data['shock_max']
        self.wr = data['wheel_radius']
        self.ww = data['wheel_width']

    @classmethod
    def from_data(cls, data: dict) -> Hardpoints:
        raise NotImplementedError
    
    @classmethod
    def link_lengths(cls, hp: Hardpoints) -> Dict[str, float]:
        raise NotImplementedError
    
    @classmethod
    def mirror_points(cls, hp: Hardpoints) -> Hardpoints:
        """Return a new Hardpoints instance with left/right points mirrored about the xz plane."""
        mirrored_data = {}
        for attr, value in hp.__dict__.items():
            if isinstance(value, np.ndarray) and value.shape == (3,):
                mirrored_data[attr] = np.array([value[0], -value[1], value[2]])
            else:
                mirrored_data[attr] = value
        return cls(**mirrored_data)

@dataclass
class DoubleAArm(Hardpoints):

    # inboard a arm points
    uf: np.ndarray          # upper front
    ur: np.ndarray          # upper rear
    lf: np.ndarray          # lower front
    lr: np.ndarray          # lower rear

    # upright joints
    ubj: np.ndarray         # upper ball joint
    lbj: np.ndarray         # lower ball joint

    # steering points
    tr_ib: np.ndarray       # tie rod inboard
    tr_ob: np.ndarray       # tie rod outboard

    # shock points
    s_loc: str              # mounting location of outboard shock point
    s_ib: np.ndarray        # shock inboard
    s_ob: np.ndarray        # shock outboard

    # pivot points
    piv_ib:  np.ndarray      # in-board pivot center (cv)
    piv_ob:  np.ndarray      # out-board pivot center (cv)

    # wheel points
    wc: np.ndarray          # wheel center point

    @classmethod
    def from_data(cls, data: dict) -> "DoubleAArm":
        return DoubleAArm(
            uf   =np.array(data['upper_a_arm_front']),
            ur   =np.array(data['upper_a_arm_rear']),
            lf   =np.array(data['lower_a_arm_front']),
            lr   =np.array(data['lower_a_arm_rear']),
            ubj  =np.array(data['upper_ball_joint']),
            lbj  =np.array(data['lower_ball_joint']),
            tr_ib=np.array(data['tie_rod_inboard']),
            tr_ob=np.array(data['tie_rod_outboard']),
            s_loc=str(data['shock_location']),
            s_ib =np.array(data['shock_inboard']),
            s_ob =np.array(data['shock_outboard']),
            piv_ib=np.array(data['pivot_inboard']),
            piv_ob=np.array(data['pivot_outboard']),
            wc   =np.array(data['wheel_center']),
        )

    @classmethod
    def link_lengths(cls, hp: "DoubleAArm") -> Dict[str, float]:
        return {
            "upper_front": float(np.linalg.norm(hp.ubj - hp.uf)),
            "upper_rear": float(np.linalg.norm(hp.ubj - hp.ur)),
            "lower_front": float(np.linalg.norm(hp.lbj - hp.lf)),
            "lower_rear": float(np.linalg.norm(hp.lbj - hp.lr)),
            "tie_rod": float(np.linalg.norm(hp.tr_ib - hp.tr_ob)),
            "shock_static": float(np.linalg.norm(hp.s_ib - hp.s_ob)),
            "axle_ib_ob_static": float(np.linalg.norm(hp.piv_ib - hp.piv_ob)),
            "axle_ob_wc": float(np.linalg.norm(hp.piv_ob - hp.wc)),
        }

@dataclass
class SemiTrailingLink(Hardpoints):
    
    # trailing link points
    tl_f: np.ndarray         # front trailing link mount

    # camber link points
    ucl_ib: np.ndarray       # upper camber link inboard
    ucl_ob: np.ndarray       # upper camber link outboard
    lcl_ib: np.ndarray       # lower camber link inboard
    lcl_ob: np.ndarray       # lower camber link outboard

    # shock points
    s_ib: np.ndarray         # shock inboard
    s_ob: np.ndarray         # shock outboard

    # pivot points
    piv_ib:  np.ndarray      # in-board pivot center (cv)
    piv_ob:  np.ndarray      # out-board pivot center (cv)

    # wheel points
    wc: np.ndarray           # wheel center point

    @classmethod
    def from_data(cls, data: dict) -> "SemiTrailingLink":
        return cls(
            tl_f  =np.array(data['trailing_link_front']),
            ucl_ib=np.array(data['upper_camber_link_inboard']),
            ucl_ob=np.array(data['upper_camber_link_outboard']),
            lcl_ib=np.array(data['lower_camber_link_inboard']),
            lcl_ob=np.array(data['lower_camber_link_outboard']),
            s_ib  =np.array(data['shock_inboard']),
            s_ob  =np.array(data['shock_outboard']),
            piv_ib=np.array(data['pivot_inboard']),
            piv_ob=np.array(data['pivot_outboard']),
            wc    =np.array(data['wheel_center']),
        )

    @classmethod
    def link_lengths(cls, hp: "SemiTrailingLink") -> Dict[str, float]:
        return {
            "upper_trailing_link":  float(np.linalg.norm(hp.tl_f - hp.ucl_ob)),
            "lower_trailing_link":  float(np.linalg.norm(hp.tl_f - hp.lcl_ob)),
            "upper_camber_link":    float(np.linalg.norm(hp.ucl_ib - hp.ucl_ob)),
            "lower_camber_link":    float(np.linalg.norm(hp.lcl_ib - hp.lcl_ob)),
            "shock_static":         float(np.linalg.norm(hp.s_ib - hp.s_ob)),
            "axle_ib_ob_static":    float(np.linalg.norm(hp.piv_ib - hp.piv_ob)),
            "axle_ob_wc":           float(np.linalg.norm(hp.piv_ob - hp.wc)),
        }