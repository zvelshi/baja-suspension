# default
from typing import Dict, Any, Optional, Tuple

# ours
from models.vehicle import Vehicle

class SolverBase:
    def __init__(self, vehicle: Vehicle, corner_id: Tuple[int, int]):
        self.vehicle = vehicle
        self.corner_id = corner_id

class SingleCornerSolver(SolverBase):
    """
    Solves for a single corner's position given an input state.
    """

    def solve(
        self, 
        steer_mm: Optional[float] = None, 
        travel_mm: Optional[float] = None, 
        bump_z: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        corner = self.vehicle.get_corner_from_id(self.corner_id)

        kwargs = {}

        if travel_mm is not None:
            kwargs['travel_mm'] = travel_mm

        if bump_z is not None:
            kwargs['bump_z'] = bump_z

        is_front = (self.corner_id[1] == 0)
        if is_front and steer_mm is not None:
            kwargs['steer_mm'] = steer_mm

        try:
            step_result = corner.solver.solve(**kwargs)
            return step_result
        except TypeError as e:
            print(f"Solver Error on corner {self.corner_id}: {e}")
            return None