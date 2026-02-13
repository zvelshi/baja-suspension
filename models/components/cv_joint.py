# default
from dataclasses import dataclass
from typing import List

# ours
from .joint import Joint

# third-party
import numpy as np

@dataclass
class CVJoint(Joint):
    """
    Fixed-Length CV Joint (Structural).
    Enforces that the joint stays at a fixed distance from its partner.
    """

    def __init__(self, max_angle: float = 30.0):
        super().__init__(max_angle=max_angle)

    def residual(
        self, 
        shaft_vec: np.ndarray, 
        joint_normal: np.ndarray,
        current_len: float,
        target_len: float
    ) -> List[float]:
        
        base_res = super().residual(shaft_vec, joint_normal, current_len, target_len)
        
        # 2. Length Constraint (Structural)
        # The distance MUST equal the target length.
        length_err = current_len - target_len
        
        return base_res + [length_err]

@dataclass
class PlungingCVJoint(Joint):
    """
    Plunging CV Joint.
    Allows the shaft length to vary within a specific limit (plunge_range).
    """

    def __init__(self, max_angle: float = 30.0, plunge_limit: float = 30.0):
        super().__init__(max_angle=max_angle)
        self.plunge_limit = plunge_limit

    def residual(
        self, 
        shaft_vec: np.ndarray, 
        joint_normal: np.ndarray,
        current_len: float,
        target_len: float
    ) -> List[float]:
        
        base_res = super().residual(shaft_vec, joint_normal, current_len, target_len)

        # 2. Plunge Constraint (Range)
        # We don't force it to be target_len, we just check if it's within limits.
        extension = current_len - target_len
        
        if abs(extension) > self.plunge_limit:
            # Penalize only the excess movement
            plunge_err = abs(extension) - self.plunge_limit
        else:
            plunge_err = 0.0
            
        return base_res + [plunge_err]
