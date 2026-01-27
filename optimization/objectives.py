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

class MinimumBumpSteer(OptimizationObjective):
    def calculate_cost(self, results):
        # Extract toe values
        toes = np.array([get_toe_angle(step) for step in results])

        # Cost = Max Deviation + Range
        max_abs_toe = np.max(np.abs(toes))
        toe_range = np.max(toes) - np.min(toes)
        return max_abs_toe + toe_range

    def get_scenario_type(self):
        return 'travel'
    
class ParallelSteer(OptimizationObjective):
    def calculate_cost(self, results):
        # Extract Ackermann percent values
        pcts = np.array([step['ackermann_pct'] for step in results])

        # Cost = RMSE + 10% of Spread
        rmse = np.sqrt(np.mean(pcts**2))
        spread = np.max(pcts) - np.min(pcts)
        return rmse + (spread * 0.1)
    
    def get_scenario_type(self):
        return 'ackermann'