# default
from typing import Dict

# third-party
import numpy as np

# ours
from models.hardpoints import Vehicle

def wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
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

    # Caster (double a-arm only)
    if 'ubj' in step and 'lbj' in step:
        steer_axis = step["ubj"] - step["lbj"]
        caster = np.degrees(np.arctan2(-steer_axis[0], steer_axis[2]))  # +ve = rearward
    else:
        caster = 0.0    

    return {
        "camber": camber, 
        "toe": get_toe_angle(step), 
        "caster": caster,
    }

def ackermann_percent(step: Dict, vehicle: Vehicle, steer_mm: float) -> float:
    toe_l, toe_r = None, None

    # Iterate through corners to find Left and Right Toe
    toe_l = get_toe_angle(step["corners"][0])
    toe_r = get_toe_angle(step["corners"][1])

    if toe_l is None or toe_r is None:
        return 0.0
    
    # Determine Turning Direction
    if abs(toe_l) < 0.1 and abs(toe_r) < 0.1:
        return 0.0

    if steer_mm < 0: # Turning Left
        actual_inner = toe_l
        outer = toe_r
    else: # Turning Right
        actual_inner = toe_r
        outer = toe_l

    # Calculate Ideal Ackermann
    # Track & Wheelbase
    wb = abs(vehicle.front_left.hardpoints.wc[0] - vehicle.rear_left.hardpoints.wc[0])
    trw = abs(vehicle.front_left.hardpoints.wc[1] - vehicle.front_right.hardpoints.wc[1])

    # Force Absolute Magnitudes for Geometry Math
    abs_inner = abs(actual_inner)
    abs_outer = abs(outer)

    # Ideal ackermann angle
    outer_rad = np.deg2rad(abs_outer)
    cot_outer = 1.0 / np.tan(outer_rad)
    
    # cot(ideal) = cot(outer) - (track / wheelbase)
    cot_ideal_inner = cot_outer - (trw / wb)
    
    # Handle parallel steering edge case (cot = 0)
    if cot_ideal_inner == 0:
         ideal_inner = 90.0
    else:
         ideal_inner_rad = np.arctan(1.0 / cot_ideal_inner)
         ideal_inner = np.rad2deg(ideal_inner_rad)

    # Calculate Percent based on magnitudes
    actual_delta = abs_inner - abs_outer
    ideal_delta = ideal_inner - abs_outer

    # Avoid division by zero if ideal geometry is parallel (ideal_delta ~ 0)
    if abs(ideal_delta) < 0.001:
        return 0.0
    
    return (actual_delta / ideal_delta) * 100.0

def get_toe_angle(step: Dict) -> float:
        n = step["wheel_axis"] 

        # Identify side
        is_left = step["wc"][1] < 0

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

        return toe