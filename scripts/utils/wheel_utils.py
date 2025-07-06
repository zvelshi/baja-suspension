# default
from typing import Dict

# third-party
import numpy as np

# ours

def wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return camber, toe and caster angles [deg] from one kinematic step."""

    # camber
    dz = step["W_Zp"][2] - step["W_Zm"][2]          # ≈ 2·radius
    dy = step["W_Zp"][1] - step["W_Zm"][1]          # lateral displacement caused by camber
    camber = np.degrees(np.arctan2(dy, dz))         # −ve = top inwards (convention)

    # toe
    dx    = step["W_Xp"][0] - step["W_Xm"][0]       # ≈ 2·radius
    dy_xy = step["W_Xp"][1] - step["W_Xm"][1]       # lateral offset at the rim
    toe = np.degrees(np.arctan2(dy_xy, dx))         # +ve = toe-in

    # caster (front only)
    if 'ubj' in step and 'lbj' in step:
        steer_axis = step["ubj"] - step["lbj"]
        caster = np.degrees(np.arctan2(-steer_axis[0], steer_axis[2]))  # +ve = rearward
    else:
        caster = None

    return {
        "camber": camber, 
        "toe": toe, 
        "caster": caster,
    }