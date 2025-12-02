from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict

import numpy as np

@dataclass
class Joint:
    """
    Base joint.

    ext_length:
        Nominal distance from joint centre to its 'outer' point.
    max_angle:
        Optional limit on the angle between the axle direction and
        the vector from joint centre to extremity (rad). If None,
        angle isn't limited, but we still try to keep it small.
    """
    ext_length: float
    max_angle: Optional[float] = None

    def _angle_residual(
        self,
        joint_pos: np.ndarray,
        extremity_pos: np.ndarray,
        axle_dir: np.ndarray,
    ) -> float:
        v = extremity_pos - joint_pos
        v_norm = np.linalg.norm(v)
        a_norm = np.linalg.norm(axle_dir)

        if v_norm < 1e-9 or a_norm < 1e-9:
            return 0.0

        v_hat = v / v_norm
        a_hat = axle_dir / a_norm

        cosang = np.clip(np.dot(v_hat, a_hat), -1.0, 1.0)
        angle = np.arccos(cosang) # radians

        if self.max_angle is None:
            # Try to keep the joint roughly aligned with the axle
            return angle

        # Only penalise violation beyond max_angle
        return max(0.0, angle - self.max_angle)

    def _length_residual(
        self,
        joint_pos: np.ndarray,
        extremity_pos: np.ndarray,
    ) -> float:
        v = extremity_pos - joint_pos
        v_norm = np.linalg.norm(v)
        return v_norm - self.ext_length

    def residual(
        self,
        joint_pos: np.ndarray,
        extremity_pos: np.ndarray,
        axle_dir: np.ndarray,
    ) -> float:
        """
        Default behaviour: combination of angle and length residuals.
        Derived joints can add more terms.
        """
        angle_err = self._angle_residual(joint_pos, extremity_pos, axle_dir)
        length_err = self._length_residual(joint_pos, extremity_pos)
        # single scalar – least_squares will minimise this
        return angle_err + length_err

@dataclass
class CVJoint(Joint):
    """
    Constant-velocity joint.

    For now this is just the base behaviour: angle + length residual.
    """

    def residual(
        self,
        joint_pos: np.ndarray,
        extremity_pos: np.ndarray,
        axle_dir: np.ndarray,
    ) -> float:
        return super().residual(joint_pos, extremity_pos, axle_dir)

@dataclass
class PlungingCVJoint(Joint):
    """
    Plunging CV joint.

    plunge_axis:
        Axis along which the extremity is allowed to move relative
        to the joint centre ('x', 'y', or 'z'). Motion perpendicular
        to this axis is penalised.
    """
    plunge_axis: Literal["x", "y", "z"] = "x"

    def residual(
        self,
        joint_pos: np.ndarray,
        extremity_pos: np.ndarray,
        axle_dir: np.ndarray,
    ) -> float:
        base = super().residual(joint_pos, extremity_pos, axle_dir)

        delta = extremity_pos - joint_pos
        if self.plunge_axis == "x":
            # Only x is allowed to change: penalise y,z
            perp = np.array([0.0, delta[1], delta[2]])
        elif self.plunge_axis == "y":
            perp = np.array([delta[0], 0.0, delta[2]])
        else: # "z"
            perp = np.array([delta[0], delta[1], 0.0])

        plunge_err = np.linalg.norm(perp)

        return base + plunge_err

@dataclass
class UJoint(Joint):
    """
    Placeholder for a future U-joint implementation.
    """
    def residual(
        self,
        joint_pos: np.ndarray,
        extremity_pos: np.ndarray,
        axle_dir: np.ndarray,
    ) -> float:
        # For now: just behave like a basic joint (angle + length)
        return super().residual(joint_pos, extremity_pos, axle_dir)

@dataclass
class Axle:
    """
    Simple axle connecting two joints.

    length:
        Nominal distance between joint1 centre and joint2 centre.
    """
    joint1: Joint
    joint2: Joint
    length: float

    def constraints(
        self,
        joint1_pos: np.ndarray,
        joint1_ext: np.ndarray,
        joint2_pos: np.ndarray,
        joint2_ext: np.ndarray,
    ) -> Dict[str, float]:
        axis_vec = joint2_pos - joint1_pos
        axis_norm = np.linalg.norm(axis_vec)
        if axis_norm < 1e-9:
            axle_dir = np.zeros(3)
        else:
            axle_dir = axis_vec / axis_norm

        length_err = axis_norm - self.length
        j1_err = self.joint1.residual(joint1_pos, joint1_ext, axle_dir)
        j2_err = self.joint2.residual(joint2_pos, joint2_ext, axle_dir)

        return {
            "length": length_err,
            "joint1": j1_err,
            "joint2": j2_err,
        }
