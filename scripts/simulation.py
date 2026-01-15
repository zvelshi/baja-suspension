# default
from typing import Dict

# third-party
import numpy as np

# ours
from models.hardpoints import Vehicle, Corner

class Simulation:
    def __init__(self, vehicle: Vehicle, sim_config: Dict):
        self.vehicle = vehicle
        self.config: Dict = sim_config

    def run(self, **kwargs):
        raise NotImplementedError

class WheelAttitudeSimulation(Simulation):
    def run(self, corner: Corner):
        solver = corner.solver
        steps = []
        count = self.config['SIM_STEPS']

        if corner.pos[1] == 0:

            # front
            mode = self.config["SIMULATION"]

            if mode == "steer":
                print("here3")
                steer_vals = np.linspace(self.config['STEER']['MIN'], self.config['STEER']['MAX'], count)
                bump_vals = [0.0] * count

                for steer, bump in zip(steer_vals, bump_vals):
                    step = solver.solve(steer_mm=steer, bump_z=bump)
                    if step:
                        steps.append(step)

                return steps

            elif mode == "travel":
                travel_vals = np.linspace(self.config['TRAVEL']['MIN'], self.config['TRAVEL']['MAX'], count)

                for travel in travel_vals:
                    step = solver.solve(travel_mm=travel)
                    if step:
                        steps.append(step)

            elif mode == "steer_travel":
                steer_vals = np.linspace(self.config['STEER']['MIN'], self.config['STEER']['MAX'], count)
                travel_vals = np.linspace(self.config['TRAVEL']['MIN'], self.config['TRAVEL']['MAX'], count)

                for travel, steer in zip(travel_vals, steer_vals):
                    step = solver.solve(travel_mm=travel, steer_mm=steer)
                    if step:
                        steps.append(step)
        else:
            # rear
            if "steer" in self.config["SIMULATION"]: 
                print("Warning: Steer input ignored for rear corner simulation.")
                return 

            travel_vals = np.linspace(self.config['TRAVEL']['MIN'], self.config['TRAVEL']['MAX'], count)
            for travel in travel_vals:
                step = solver.solve(travel_mm=travel)
                if step:
                    steps.append(step)

        return steps

class AckermannSimulation(Simulation):
    def run(self):
        results = []

        # Identify front corners
        front_corners = [self.vehicle.front_left, self.vehicle.front_right]

        if not front_corners:
            return results

        bump_z = 0.0

        # Generate sweep values
        steer_steps = self.config['SIM_STEPS']
        steer_values = np.linspace(self.config['STEER']['MIN'], self.config['STEER']['MAX'], steer_steps, True)

        for steer_mm in steer_values:
            step_data = {
                "input": {
                    "steer_mm": steer_mm,
                    "bump_z": bump_z
                },
                "corners": {}
            }
            for corner in front_corners:
                solution = corner.solver.solve(bump_z=bump_z, steer_mm=steer_mm)
                if solution:
                    key = corner.pos[0]
                    step_data["corners"][key] = solution
            results.append(step_data)
        return results