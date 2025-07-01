from __future__ import annotations

# default
import yaml
from dataclasses import dataclass
from typing import Dict, Tuple

# third-party
import numpy as np

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
    shock_chassis: np.ndarray   # shock chassis
    shock_a_arm: np.ndarray     # shock a arm
    shock_min: float            # shock min length
    shock_max: float            # shock max length

    # wheel points
    wc: np.ndarray          # wheel center point
    wr: float               # wheel radius
    ww: float               # wheel width
    static_camber: float    # static camber angle in degrees

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
            wr=data['wheel_radius'],
            ww=data['wheel_width'],
            static_camber=data['static_camber'],
            shock_min=data['shock_min'],
            shock_max=data['shock_max'],
        )
    
    @classmethod
    def link_lengths(self, hp: "DoubleAArmHardpoints") -> Dict[str, float]:
        """Calculate link lengths based on hardpoints."""

        return {
            "upper_front": np.linalg.norm(hp.bju - hp.uf),
            "upper_rear": np.linalg.norm(hp.bju - hp.ur),
            "lower_front": np.linalg.norm(hp.bjl - hp.lf),
            "lower_rear": np.linalg.norm(hp.bjl - hp.lr),
            "tie_rod": np.linalg.norm(hp.tr_chassis - hp.tr_upright),
            "shock_static": np.linalg.norm(hp.shock_chassis - hp.shock_a_arm),
        }