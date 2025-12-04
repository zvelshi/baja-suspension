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

        # fix: front (pos[1] == 0) -> travel & steer; rear -> travel only
        if corner.pos[1] == 0:
            travel_values = (
                np.linspace(self.config['TRAVEL']['MIN'], self.config['TRAVEL']['MAX'], self.config['SIM_STEPS'], True)
                if self.config['TRAVEL']['ENABLE'] else [self.config['TRAVEL']['VALUE']] * self.config['SIM_STEPS']
            )
            steer_values = (
                np.linspace(self.config['STEER']['MIN'], self.config['STEER']['MAX'], self.config['SIM_STEPS'], True)
                if self.config['STEER']['ENABLE'] else [self.config['STEER']['VALUE']] * self.config['SIM_STEPS']
            )
            for travel, steer in zip(travel_values, steer_values):
                step = solver.solve(travel_mm=travel, steer_mm=steer)
                if step:
                    steps.append(step)
        else:
            travel_values = (
                np.linspace(self.config['TRAVEL']['MIN'], self.config['TRAVEL']['MAX'], self.config['SIM_STEPS'], True)
                if self.config['TRAVEL']['ENABLE'] else [self.config['TRAVEL']['VALUE']]
            )
            for travel in travel_values:
                step = solver.solve(travel_mm=travel)
                if step:
                    steps.append(step)

        return steps

class JackingSimulation(Simulation):
    def run(self, half: str):
        if half not in ['front', 'rear']:
            raise ValueError("half must be 'front' or 'rear'")

        # TODO: implement jacking response (left/right)
        # For now, just return an empty list to keep interface consistent.
        return []