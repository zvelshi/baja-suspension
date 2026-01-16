# default
from typing import Dict, Tuple

# third-party
import numpy as np

def get_wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculates Camber, Toe, and Caster from a simulation step.
    Expects step to contain 'wheel_axis', 'wc', 'ubj', 'lbj' (optional).
    """
    # Get the wheel's rotation axis (wheel y-axis)
    n = step["wheel_axis"] 
    
    # Identify side based on Wheel Center Y (Left is +Y in some ISO, but here Left is Y>0 or Y<0 depending on convention)
    # Based on your files: Left WC Y < 0 seems to be the convention in wheel_utils.py? 
    # Let's stick to the logic provided in your original wheel_utils.py:
    # "is_left = step['wc'][1] < 0"
    is_left = step["wc"][1] < 0
    
    # Camber - projection of wheel y onto global z
    camber_rad = np.arcsin(n[2])
    camber = np.rad2deg(camber_rad)
    
    # Sign correction: top-in is negative
    if not is_left:
        camber = -camber

    # Toe
    toe = get_toe_angle(step)

    # Caster (Double A-Arm only check)
    if 'ubj' in step and 'lbj' in step:
        steer_axis = step["ubj"] - step["lbj"]
        # angle between steer axis and vertical in side view
        caster = np.degrees(np.arctan2(-steer_axis[0], steer_axis[2]))  # +ve = rearward
    else:
        caster = 0.0    

    return {
        "camber": camber, 
        "toe": toe, 
        "caster": caster,
    }

def get_toe_angle(step: Dict) -> float:
    n = step["wheel_axis"] 
    is_left = step["wc"][1] < 0

    # Toe - angle of the wheel heading in the XY plane    
    # Project n onto XY plane
    n_xy = np.array([n[0], n[1]])
    norm = np.linalg.norm(n_xy)
    if norm < 1e-9:
        return 0.0
    n_xy /= norm
    
    # Angle relative to lateral axis (0, 1)
    # Deviation from Y-axis
    toe_rad = np.arctan2(n_xy[0], n_xy[1]) 
    
    # Sign Correction: toe-in is positive
    toe = np.rad2deg(toe_rad)
    if not is_left:
        toe = -toe

    return toe

def calculate_ackermann_percentage(
    inner_toe: float, 
    outer_toe: float, 
    track_width: float, 
    wheelbase: float
) -> float:
    """
    Calculates Ackermann percentage based on inner and outer toe angles.
    """
    # Force absolute magnitudes
    inner = abs(inner_toe)
    outer = abs(outer_toe)

    if outer < 0.001:  
        return 0.0

    # Ideal Inner (100% Ackermann)
    # cot(inner) = cot(outer) - (tw / wb)
    outer_rad = np.deg2rad(outer)
    cot_outer = 1.0 / np.tan(outer_rad)
    cot_ideal_inner = cot_outer - (track_width / wheelbase)
    
    ideal_inner_rad = np.arctan(1.0 / cot_ideal_inner)
    ideal_inner = np.rad2deg(ideal_inner_rad)

    # Delta (Turn relative to outer)
    actual_delta = inner - outer
    ideal_delta = ideal_inner - outer

    if abs(ideal_delta) < 0.001:
        return 0.0
        
    return (actual_delta / ideal_delta) * 100.0