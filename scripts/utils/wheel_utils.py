# default
from typing import Dict

# third-party
import numpy as np

def wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculates Toe and Camber using the Wheel Axis vector directly.
    """

    # Get the wheel's rotation axis (wheel y-axis)
    n = step["wheel_axis"] 
    
    # Identify side
    is_left = step["wc"][1] < 0
    
    # Camber - projection of wheel y onto global z
    camber_rad = np.arcsin(n[2])
    camber = np.rad2deg(camber_rad)
    
    # Sign correction: top-in is negative
    if not is_left:
        camber = -camber

    # Toe - angle of the wheel heading in the XY plane    
    # Project n onto XY plane
    n_xy = np.array([n[0], n[1]])
    n_xy /= np.linalg.norm(n_xy)
    
    # Angle relative to lateral axis (0, 1)
    toe_rad = np.arctan2(n_xy[0], n_xy[1]) # deviation from Y
    
    # Sign Correction: toe-in is positive
    toe = np.rad2deg(toe_rad)
    if not is_left:
        toe = -toe

    # Caster (double a-arm only)
    if 'ubj' in step and 'lbj' in step:
        steer_axis = step["ubj"] - step["lbj"]
        caster = np.degrees(np.arctan2(-steer_axis[0], steer_axis[2]))  # +ve = rearward
    else:
        caster = 0.0    

    return {
        "camber": camber, 
        "toe": toe, 
        "caster": caster,
    }