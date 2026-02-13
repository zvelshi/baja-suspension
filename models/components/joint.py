# default
from dataclasses import dataclass
from typing import List

# third-party
import numpy as np

@dataclass
class Joint:
    """
    Base Joint Class.
    Handles the angle constraints (checking if the shaft bends too far).
    """
    
    def __init__(self, max_angle: float = 30.0):
        self.max_angle = np.deg2rad(max_angle)

    def _angle_violation(self, shaft_vec: np.ndarray, joint_normal: np.ndarray) -> float:
        """
        Calculates how far the shaft has bent past the max operating angle.
        """

        s_norm = np.linalg.norm(shaft_vec)
        n_norm = np.linalg.norm(joint_normal)
        
        if s_norm < 1e-9 or n_norm < 1e-9:
            return 0.0
            
        s_hat = shaft_vec / s_norm
        n_hat = joint_normal / n_norm
        
        # Dot product for angle
        dot = np.clip(np.dot(s_hat, n_hat), -1.0, 1.0)
        angle = np.arccos(dot)
        
        # Penalize only if we exceed max_angle
        return max(0.0, angle - self.max_angle)

    def residual(
        self, 
        shaft_vec: np.ndarray, 
        joint_normal: np.ndarray,
        current_len: float,
        target_len: float
    ) -> List[float]:
        """
        Returns a list of constraints. 
        Base implementation only checks angles. Subclasses add length constraints.
        """

        # 1. Angle Constraint
        angle_err = self._angle_violation(shaft_vec, joint_normal)
        return [angle_err]