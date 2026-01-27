# default
from typing import Dict, Tuple

# third-party
import numpy as np

def get_wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    # Camber
    camber = get_camber_angle(step)
    
    # Toe
    toe = get_toe_angle(step)

    # Caster
    if 'ubj' in step and 'lbj' in step:
        steer_axis = step["ubj"] - step["lbj"]
        # Angle between steer axis and vertical in side view
        caster = -1*np.degrees(np.arctan2(-steer_axis[0], steer_axis[2]))
    else:
        caster = 0.0    

    return {
        "camber": camber, 
        "toe": toe, 
        "caster": caster,
    }

def get_camber_angle(step: Dict) -> float:
    n = step["wheel_axis"]

    # Camber - inclination of the wheel plane to vertical
    camber_rad = np.arcsin(n[2])
    camber = np.rad2deg(camber_rad)

    return -camber

def get_toe_angle(step: Dict) -> float:
    n = step["wheel_axis"] 
    is_left = step["wc"][1] < 0

    # Toe - inclination of the wheel plane to longitudinal axis
    n_xy = np.array([n[0], n[1]])
    norm = np.linalg.norm(n_xy)
    if norm < 1e-9: 
        return 0.0
    n_xy /= norm

    if is_left:
        n_xy[1] = -n_xy[1]

    toe_rad = np.arctan2(n_xy[0], n_xy[1]) 
    return np.rad2deg(toe_rad)
 
def calculate_ackermann_percentage(
    inner_toe: float, 
    outer_toe: float, 
    track_width: float, 
    wheelbase: float
) -> float:

    theta_3 = abs(inner_toe)  # Actual Inner
    theta_4 = abs(outer_toe)  # Actual Outer

    # Calculate centerline angle - this is the "requested steer angle"
    avg_angle = (theta_3 + theta_4) / 2.0
    
    # Calculate ideal angles based on centerline
    # cot(angle) = 1/tan(angle)
    cot_center = 1.0 / np.tan(np.deg2rad(avg_angle))
    
    # Half-width ratio for geometry calc
    hw_ratio = (track_width / 2.0) / wheelbase
    
    # Handle ideal inner angle
    if (cot_center - hw_ratio) == 0:
        theta_1 = 90.0
    else:
        theta_1 = np.rad2deg(np.arctan(1.0 / (cot_center - hw_ratio)))

    theta_2 = np.rad2deg(np.arctan(1.0 / (cot_center + hw_ratio)))

    return ((theta_3 - theta_4) / (theta_1 - theta_2)) * 100.0