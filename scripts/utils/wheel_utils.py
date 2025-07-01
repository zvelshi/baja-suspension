# default
from typing import Dict

# third-party
import numpy as np

# ours

def wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return camber, toe and caster angles [deg] from one kinematic step."""

    # camber
    dz = step["W_Zp"][2] - step["W_Zm"][2]          # z difference (should be ≈ 2·radius)
    dx = step["W_Zp"][0] - step["W_Zm"][0]          # x difference → tilt
    camber = np.degrees(np.arctan2(dx, dz))         # −ve = top inwards (convention)

    # toe
    dx_xy = step["W_Xp"][0] - step["W_Xm"][0]       # x difference (≈ 2·radius)
    dy_xy = step["W_Xp"][1] - step["W_Xm"][1]       # y difference → toe-in/out
    toe = np.degrees(np.arctan2(dy_xy, dx_xy))      # +ve = toe-in

    # caster
    steer_axis = step["upper_ball_joint"] - step["lower_ball_joint"]
    caster = np.degrees(np.arctan2(-steer_axis[0], steer_axis[2]))  # +ve = rearward

    return {
        "camber": camber, 
        "toe": toe, 
        "caster": caster
    }