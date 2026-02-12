# default
from abc import ABC, abstractmethod

# third-party
import numpy as np

# ours
from utils.geometry import get_toe_angle

class OptimizationObjective(ABC):
    @abstractmethod
    def calculate_cost(self, results: list) -> float:
        """Returns a scalar cost for a given simulation run."""
        pass

    @abstractmethod
    def get_scenario_type(self) -> str:
        """Returns the key of the scenario class to run ('steer', 'travel', 'steer_travel', 'ackermann')."""
        pass

    @property
    def name(self):
        """Helper to get class name for logging/plotting."""
        return self.__class__.__name__ 

class MinimumBumpSteer(OptimizationObjective):
    def calculate_cost(self, results):
        toes = np.array([get_toe_angle(step) for step in results])
        max_abs_toe = np.max(np.abs(toes))
        toe_range = np.max(toes) - np.min(toes)
        return (max_abs_toe + toe_range)/150.0

    def get_scenario_type(self):
        return 'travel'
    
class ParallelSteer(OptimizationObjective):
    def calculate_cost(self, results):
        pcts = np.array([step['ackermann_pct'] for step in results])
        rmse = np.sqrt(np.mean(pcts**2))
        return rmse/1400.0
    
    def get_scenario_type(self):
        return 'ackermann'