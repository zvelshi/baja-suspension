from __future__ import annotations

# default
from typing import Dict, Tuple

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from models.corners.double_a_arm import DoubleAArmNumeric
from models.corners.semi_trailing_link import SemiTrailingLinkNumeric
from models.components.shock import Shock
from utils.misc import log_to_file

class Vehicle:
    nickname: str

    def __init__(self, data: Dict = {}):
        self.nickname = list(data.keys())[0]
        vehicle_data = data[self.nickname]

        s_mass = vehicle_data.get('sprung_mass', {'fl': 0, 'fr': 0, 'rl': 0, 'rr': 0})
        u_mass = vehicle_data.get('unsprung_mass', {'fl': 0, 'fr': 0, 'rl': 0, 'rr': 0})
                            
        self.front_left  = Corner(vehicle_data, (0, 0), s_mass['fl'], u_mass['fl'])
        self.front_right = Corner(vehicle_data, (1, 0), s_mass['fr'], u_mass['fr'])
        self.rear_left   = Corner(vehicle_data, (0, 1), s_mass['rl'], u_mass['rl'])
        self.rear_right  = Corner(vehicle_data, (1, 1), s_mass['rr'], u_mass['rr'])

        self.total_sprung_mass, self.sprung_bias_f = self._get_mass()
        log_to_file(f"Initialized Vehicle '{self.nickname}'")

        self.cog = self._calculate_cog()
        log_to_file(f"Calculated COG at (x={self.cog[0]:.2f}, y={self.cog[1]:.2f}, z={self.cog[2]:.2f})")

    def _get_mass(self) -> Tuple[float, float]:
        corners = [self.front_left, self.front_right, self.rear_left, self.rear_right]

        total_s = sum(c.sprung_mass for c in corners)
        front_s = self.front_left.sprung_mass + self.front_right.sprung_mass

        bias_f = (front_s / total_s)
        return total_s, bias_f

    def _calculate_cog(self) -> Tuple[float, float, float]:
        corners = [self.front_left, self.front_right, self.rear_left, self.rear_right]

        # Using Sprung Mass
        sum_mx = sum(c.sprung_mass * c.hardpoints.wc[0] for c in corners)
        sum_my = sum(c.sprung_mass * c.hardpoints.wc[1] for c in corners)

        cg_x = sum_mx / self.total_sprung_mass
        cg_y = sum_my / self.total_sprung_mass

        return (cg_x, cg_y, 300)

    def run_simulation(self, simulation_class, **kwargs):
        simulation = simulation_class(self, kwargs.get("config", {}))
        return simulation.run()

    def get_corner_from_id(self, id) -> Corner:
        if id == [0, 0]: return self.front_left
        if id == [1, 0]: return self.front_right
        if id == [0, 1]: return self.rear_left
        if id == [1, 1]: return self.rear_right
        raise ValueError(f"Invalid corner_id: {id}")

class Corner:
    """
         (0, 0) _________ (1, 0)
                |       |
                |       |
                |       |
                |       |
                |       |
         (0, 1) |_______| (1, 1)
    """
    def __init__(self, data: Dict, id: Tuple[int, int], sprung_mass: float, unsprung_mass: float):
        self.id = id
        self.sprung_mass = sprung_mass
        self.unsprung_mass = unsprung_mass
        self.total_mass = sprung_mass + unsprung_mass

        if self.id[1] == 0:
            corner_data = data['front']
            hp = DoubleAArm.from_data(data=data['front'])
        else:
            corner_data = data['rear']
            hp = SemiTrailingLink.from_data(data=data['rear'])

        if self.id[0] == 0:  # left side -> mirror across y-axis
            hp = type(hp).mirror_points(hp)

        hp._fill_vehicle_properties(data=data)

        self.hardpoints = hp
        self.solver = DoubleAArmNumeric(hp) if isinstance(hp, DoubleAArm) else SemiTrailingLinkNumeric(hp)
        self.shock = Shock.from_config(corner_data, data['shock_max'], data['shock_min'])