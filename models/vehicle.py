from __future__ import annotations

# default
from typing import Dict, Tuple

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from models.corners.double_a_arm import DoubleAArmNumeric
from models.corners.semi_trailing_link import SemiTrailingLinkNumeric

class Vehicle:
    nickname: str

    def __init__(self, data: Dict = {}):
        self.nickname = list(data.keys())[0]
        vehicle_data = data[self.nickname]

        # (left/right, front/rear)
        self.front_left  = Corner(vehicle_data, (0, 0))
        self.front_right = Corner(vehicle_data, (1, 0))
        self.rear_left   = Corner(vehicle_data, (0, 1))
        self.rear_right  = Corner(vehicle_data, (1, 1))

    def run_simulation(self, simulation_class, **kwargs):
        simulation = simulation_class(self, kwargs.get("config", {}))
        return simulation.run(**kwargs)
    
    def get_corner_from_id(self, id) -> Corner:
        if id == [0, 0]:
            return self.front_left
        elif id == [1, 0]:
            return self.front_right
        elif id == [0, 1]:
            return self.rear_left
        elif id == [1, 1]:
            return self.rear_right
        else:
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
    def __init__(self, data: Dict, id: Tuple[int, int]):
        self.id = id  # (left/right, front/rear)

        if self.id[1] == 0:
            hp = DoubleAArm.from_data(data=data['front'])
        else:
            hp = SemiTrailingLink.from_data(data=data['rear'])

        if self.id[0] == 0:  # left side -> mirror across y-axis
            hp = type(hp).mirror_points(hp)

        hp._fill_vehicle_properties(data=data)

        self.hardpoints = hp
        self.solver = DoubleAArmNumeric(hp) if isinstance(hp, DoubleAArm) else SemiTrailingLinkNumeric(hp)