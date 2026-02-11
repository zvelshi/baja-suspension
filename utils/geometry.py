# default
from typing import Dict

# third-party
import numpy as np

def get_wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "camber": get_camber_angle(step), 
        "toe": get_toe_angle(step), 
        "caster": get_caster_angle(step),
    }

def get_camber_angle(step: Dict) -> float:
    n = step["wheel_axis"]
    camber_rad = np.arcsin(n[2])
    return -np.rad2deg(camber_rad)

def get_toe_angle(step: Dict) -> float:
    n = step["wheel_axis"]
    toe_rad = np.arcsin(n[0])
    return -np.rad2deg(toe_rad)

def get_caster_angle(step: Dict) -> float:
    v = step["ubj"] - step["lbj"]
    return np.rad2deg(np.arctan2(v[0], v[2]))
 
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