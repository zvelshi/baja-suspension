# default
from typing import List, Dict, Any

# third-party
import numpy as np
import scipy.optimize

# ours
from .solvers import SingleCornerSolver
from utils.geometry import calculate_ackermann_percentage, get_toe_angle
from utils.misc import log_to_file

class Scenario:
    def run(self) -> List[Any]:
        raise NotImplementedError

class SuspensionSweep(Scenario):
    """
    Sweeps a single corner through Travel AND/OR Steer.
    """

    def __init__(self, vehicle, config):
        self.config = config

        self.corner_id = [0, 0]
        if config["HALF"] == 'rear':
            self.corner_id[1] = 1
        elif config["SIDE"] == 'right':
            self.corner_id[0] = 1

        self.solver = SingleCornerSolver(vehicle, self.corner_id)

    def run(self) -> List[Dict]:
        steps = []
        count = self.config['SIM_STEPS']
        
        # Helper to generate ranges
        def get_range(key):
            return np.linspace(self.config[key]['MIN'], self.config[key]['MAX'], count)

        sim_type = self.config["SIMULATION"]
        log_to_file(f"Starting SuspensionSweep: {sim_type} on corner {self.corner_id}")

        if sim_type == "steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, bump_z=0.0)
                if res: 
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Steer sweep step failed at {s:.2f}mm")

        elif sim_type == "travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=0.0, travel_mm=t)
                if res: 
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Travel sweep step failed at {t:.2f}mm")

        elif sim_type == "steer_travel":
             s_vals = get_range('STEER')
             t_vals = get_range('TRAVEL')
             for s, t in zip(s_vals, t_vals):
                res = self.solver.solve(steer_mm=s, travel_mm=t)
                if res: 
                    res['x_val'] = t # Default X-axis to travel for combined sweeps
                    res['x_label'] = "Shock Travel [mm] (with Steer)"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Combined sweep step failed at steer={s:.2f}, travel={t:.2f}")
                  
        return steps

class AckermannScenario(Scenario):
    """
    Simulates both front wheels steering to calculate Ackermann percentage.
    """

    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config
        self.l_solver = SingleCornerSolver(vehicle, corner_id=[0, 0])
        self.r_solver = SingleCornerSolver(vehicle, corner_id=[1, 0])

    def run(self) -> List[Dict]:
        results = []
        log_to_file("Starting Ackermann Analysis...")

        wb = abs(self.vehicle.front_left.hardpoints.wc[0] - self.vehicle.rear_left.hardpoints.wc[0])
        tw = abs(self.vehicle.front_left.hardpoints.wc[1] - self.vehicle.front_right.hardpoints.wc[1])

        steer_steps = np.linspace(
            self.config['STEER']['MIN'], 
            self.config['STEER']['MAX'], 
            self.config['SIM_STEPS']
        )

        for steer in steer_steps:
            input = steer if steer != 0 else 0.001

            left = self.l_solver.solve(steer_mm=input, bump_z=0.0)
            right = self.r_solver.solve(steer_mm=input, bump_z=0.0)

            if left and right:
                toe_l = get_toe_angle(left)
                toe_r = get_toe_angle(right)

                if steer < 0: 
                    ack_pct = calculate_ackermann_percentage(toe_l, toe_r, tw, wb)
                else:
                    ack_pct = calculate_ackermann_percentage(toe_r, toe_l, tw, wb)

                results.append({
                    "input": input,
                    "left": left,
                    "right": right,
                    "ackermann_pct": ack_pct
                })
            else:
                log_to_file(f"[WARN] Ackermann step failed at input {input:.2f}. Left={bool(left)}, Right={bool(right)}")
                
        return results

class DynamicScenario(Scenario):
    """
    Simulates the vehicle driving over a single sinusoidal terrain profile.
    CoG height is fixed at the design height, but moves longitudinally.
    """
    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config

        self.solvers = {
            'fl': SingleCornerSolver(vehicle, [0, 0]),
            'fr': SingleCornerSolver(vehicle, [1, 0]),
            'rl': SingleCornerSolver(vehicle, [0, 1]),
            'rr': SingleCornerSolver(vehicle, [1, 1]),
        }

    def _get_terrain_height_mm(self, x_mm: float) -> float:
        """Calculates terrain Z using a single sine wave."""
        t_cfg = self.config['TERRAIN']
        x_m = x_mm / 1000.0
        z_mm = t_cfg['AMPLITUDE'] * np.sin(2 * np.pi * t_cfg['FREQUENCY'] * x_m)
        return z_mm

    def run(self) -> List[Dict]:
        results = []
        sol_dt = self.config['SOL_DT']
        viz_dt = self.config['VIZ_DT']

        total_time = self.config['DURATION']
        velocity = -self.config['VELOCITY'] * 1000.0 # [mm/s]

        sub_steps = int(viz_dt / sol_dt)
        total_render_frames = int(total_time / viz_dt)
        print(f"--- Running Kinematic Sim ---")

        for i in range(total_render_frames):
            t_render = i * viz_dt
            car_x = velocity * t_render

            for _ in range(sub_steps):
                # Physics 
                pass
            
            step_data = {
                'time': t_render,
                'x_pos': car_x,
                'cog_z': self.vehicle.cog[2],
                'cog_x': self.vehicle.front_left.hardpoints.wc[0] + (self.vehicle.cog[0] * self.vehicle.sprung_bias_f),
                'corners': {}
            }
            
            for name, solver in self.solvers.items():
                corner = self.vehicle.get_corner_from_id(solver.corner_id)
                wx = car_x + corner.hardpoints.wc[0]
                ground_z = self._get_terrain_height_mm(wx)
                bump_z = ground_z - (self.vehicle.cog[2] - corner.hardpoints.wr) 

                res = solver.solve(steer_mm=0.0, bump_z=bump_z)
                if res:
                    step_data['corners'][name] = res
            
            results.append(step_data)
        return results