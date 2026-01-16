# default
from typing import List, Dict, Any

# third-party
import numpy as np

# ours
from .solvers import SingleCornerSolver
from utils.geometry import get_wheel_attitude, calculate_ackermann_percentage, get_toe_angle

class Scenario:
    def run(self) -> List[Any]:
        raise NotImplementedError

class SuspensionSweep(Scenario):
    """
    Sweeps a single corner through Travel or Steer.
    """
    def __init__(
        self, 
        vehicle, 
        corner_id: tuple[int, int],
        mode: str, 
        config: Dict
    ):
        self.solver = SingleCornerSolver(vehicle)
        self.corner_id = corner_id
        self.mode = mode # 'steer', 'travel', 'steer_travel'
        self.config = config

    def run(self) -> List[Dict]:
        steps = []
        count = self.config['SIM_STEPS']
        
        # Helper to generate ranges
        def get_range(key):
            return np.linspace(self.config[key]['MIN'], self.config[key]['MAX'], count)

        if self.mode == "steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(self.corner_id, steer_mm=s, bump_z=0.0)
                if res: 
                    steps.append(res)

        elif self.mode == "travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(self.corner_id, steer_mm=0.0, travel_mm=t)
                if res: 
                    steps.append(res)
                
        elif self.mode == "steer_travel":
             s_vals = get_range('STEER')
             t_vals = get_range('TRAVEL')
             for s, t in zip(s_vals, t_vals):
                res = self.solver.solve(self.corner_id, steer_mm=s, travel_mm=t)
                if res: 
                    steps.append(res)
                 
        return steps

class AckermannScenario(Scenario):
    """
    Simulates both front wheels steering to calculate Ackermann percentage.
    """
    def __init__(self, vehicle, config: Dict):
        self.vehicle = vehicle
        self.solver = SingleCornerSolver(vehicle)
        self.config = config

    def run(self) -> List[Dict]:
        results = []
        
        id_fl = [0, 0]
        id_fr = [1, 0]
        
        wb = abs(self.vehicle.front_left.hardpoints.wc[0] - self.vehicle.rear_left.hardpoints.wc[0])
        tw = abs(self.vehicle.front_left.hardpoints.wc[1] - self.vehicle.front_right.hardpoints.wc[1])

        steer_steps = np.linspace(
            self.config['STEER']['MIN'], 
            self.config['STEER']['MAX'], 
            self.config['SIM_STEPS']
        )

        for steer in steer_steps:
            left = self.solver.solve(id_fl, steer_mm=steer, bump_z=0.0)
            right = self.solver.solve(id_fr, steer_mm=steer, bump_z=0.0)

            if left and right:
                toe_l = get_toe_angle(left)
                toe_r = get_toe_angle(right)

                if steer < 0: 
                    ack_pct = calculate_ackermann_percentage(toe_l, toe_r, tw, wb)
                else:
                    ack_pct = calculate_ackermann_percentage(toe_r, toe_l, tw, wb)

                results.append({
                    "input": steer,
                    "left": left,
                    "right": right,
                    "ackermann_pct": ack_pct
                })
                
        return results