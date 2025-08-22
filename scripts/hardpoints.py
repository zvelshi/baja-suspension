from __future__ import annotations

# default
import yaml
from dataclasses import dataclass, field
from typing import Dict, Tuple

# third-party
import numpy as np

class Vehicle:
    nickname: str       # vehicle nickname

    shock_min: float    # compressed length of shock
    shock_max: float    # extended length of shock

    wheel_radius: float # tire radius
    wheel_width: float  # tire width

    front_type: str
    front_hp: Hardpoints

    rear_type: str
    rear_hp: Hardpoints

    def __init__(self, yml_path):
        with open(yml_path, 'r') as file:
            data = yaml.safe_load(file)

        self.nickname = list(data.keys())[0]
        vehicle = data[self.nickname]

        self.shock_min = vehicle['shock_min']
        self.shock_max = vehicle['shock_max']

        self.wheel_radius = vehicle['wheel_radius']
        self.wheel_width  = vehicle['wheel_width']

        if 'front' in vehicle:
            self.front_type = vehicle['front']['_type']
            if self.front_type == 'DoubleAArm':
                self.front_hp = DoubleAArm.from_data(data=vehicle['front'])
                self.front_hp._fill_vehicle_properties(data=vehicle)

        if 'rear' in vehicle:
            self.rear_type  = vehicle['rear']['_type']
            if self.rear_type == 'SemiTrailingLink':
                self.rear_hp = SemiTrailingLink.from_data(data=vehicle['rear'])
                self.rear_hp._fill_vehicle_properties(data=vehicle)

@dataclass
class Hardpoints:

    # shock properties
    shock_min: float = field(default=0.0, init=False)   # shock min
    shock_max: float = field(default=0.0, init=False)   # shock max

    # wheel properties
    wr: float        = field(default=0.0, init=False)   # wheel radius
    ww: float        = field(default=0.0, init=False)   # wheel width

    def _fill_vehicle_properties(self, data):
        self.shock_min = data['shock_min']
        self.shock_max = data['shock_max']
        self.wr = data['wheel_radius']
        self.ww = data['wheel_width']

    @classmethod
    def from_data(self, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def link_lengths(self, *args, **kwargs):
        raise NotImplementedError

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

    # wheel points
    wc: np.ndarray          # wheel center point

    @classmethod
    def from_data(self, data: dict) -> "DoubleAArm":
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
            wc   =np.array(data['wheel_center']),
        )
    
    @classmethod
    def link_lengths(self, hp: "DoubleAArm") -> Dict[str, float]:
        return {
            "upper_front": np.linalg.norm(hp.ubj - hp.uf),
            "upper_rear": np.linalg.norm(hp.ubj - hp.ur),
            "lower_front": np.linalg.norm(hp.lbj - hp.lf),
            "lower_rear": np.linalg.norm(hp.lbj - hp.lr),
            "tie_rod": np.linalg.norm(hp.tr_ib - hp.tr_ob),
            "shock_static": np.linalg.norm(hp.s_ib - hp.s_ob),
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
    def from_data(self, data: dict) -> "SemiTrailingLink":
        return SemiTrailingLink(
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
    def link_lengths(self, hp: "SemiTrailingLink") -> Dict[str, float]:
        return {
            "upper_trailing_link":  np.linalg.norm(hp.tl_f - hp.ucl_ob),
            "lower_trailing_link":  np.linalg.norm(hp.tl_f - hp.lcl_ob),
            "upper_camber_link":    np.linalg.norm(hp.ucl_ib - hp.ucl_ob),
            "lower_camber_link":    np.linalg.norm(hp.lcl_ib - hp.lcl_ob),
            "shock_static":         np.linalg.norm(hp.s_ib - hp.s_ob),
            "axle_ib_ob_static":    np.linalg.norm(hp.piv_ib - hp.piv_ob),
            "axle_ob_wc":           np.linalg.norm(hp.piv_ob - hp.wc),
        }