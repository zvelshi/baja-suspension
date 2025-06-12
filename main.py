# default
import time
from typing import Dict, Tuple

# third party
import numpy as np

# ours
from models.double_a_arm_numeric import DoubleAArmNumeric
from hardpoints import DoubleAArmHardpoints
from plotter import Plotter

if __name__ == "__main__":
    hp = DoubleAArmHardpoints.from_yml("hardpoints.yml")
    solver = DoubleAArmNumeric(hp)
    plotter = Plotter(hp)

    sh_len = np.linspace(-200, 400, 20, True)
    bp_len = np.linspace(-80, 120, 20, True)
    st_len = np.linspace(-20, 20, 20, True)

    for x in st_len:
        step = solver.solve_from_damper(bump_z=0, steer_mm=x)
        if step:
            plotter.update(step)

    time.sleep(5)

    for x in sh_len:
        step = solver.solve_from_damper(travel_mm=x, steer_mm=0)
        if step:
            plotter.update(step)

    plotter.display()