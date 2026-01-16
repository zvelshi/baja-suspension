# default
from abc import ABC, abstractmethod

# third-party
import numpy as np

# ours
from simulations.solvers import WheelAttitudeSolver

class OptimizationObjective(ABC):
    @abstractmethod
    def calculate_cost(self, simulation_results: list) -> float:
        """Returns a scalar cost for a given simulation run."""
        pass

    @abstractmethod
    def get_simulation_config(self) -> dict:
        """Returns the config needed to run the sim (e.g. range of travel)."""
        pass

class MinimumBumpSteer(OptimizationObjective):
    def calculate_cost(self, steps):
        # Extract toe values
        toes = np.array([step['attitude']['toe'] for step in steps])
        
        # Cost = Max Deviation + Range
        max_abs_toe = np.max(np.abs(toes))
        toe_range = np.max(toes) - np.min(toes)
        return max_abs_toe + toe_range

    def get_simulation_config(self):
        # Force the sim to run in 'travel' mode for bump steer
        return {"SIMULATION": "travel"}