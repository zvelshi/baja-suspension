# default
import time
from typing import Dict, Tuple

# third party
import numpy as np

# ours
from models.double_a_arm_numeric import DoubleAArmNumeric
from scripts.hardpoints import DoubleAArmHardpoints
from scripts.plotter import DoubleAArmPlotter, Plotter2DBase, SCALAR_CHARACTERISTIC
from scripts.utils.wheel_utils import wheel_attitude

if __name__ == "__main__":
    hp = DoubleAArmHardpoints.from_yml("hardpoints/2021.yml")
    solver = DoubleAArmNumeric(hp)
    daa_plotter  = DoubleAArmPlotter(hp)
    camber_plot  = Plotter2DBase(SCALAR_CHARACTERISTIC.CAMBER)
    caster_plot  = Plotter2DBase(SCALAR_CHARACTERISTIC.CASTER)
    toe_plot     = Plotter2DBase(SCALAR_CHARACTERISTIC.TOE)

    sh_len = np.linspace(-200, 400, 350, True)
    bp_len = np.linspace(-80, 120, 20, True)
    st_len = np.linspace(-20, 20, 3, True)

    # for x in st_len:
    #     step = solver.solve_from_damper(bump_z=0, steer_mm=x)
    #     if step:
    #         plotter.update(step)

    # time.sleep(5)
    # for y in st_len:
    for x in sh_len:
        step = solver.solve(travel_mm=x, steer_mm=0)
        if step:
            # 3d
            daa_plotter.update(step)

            # 2d
            att = wheel_attitude(step)
            camber_plot.update(att)
            caster_plot.update(att)
            toe_plot.update(att)
    solver.reset()

    daa_plotter.display()
    camber_plot.display()
    caster_plot.display()
    toe_plot.display()

# camber through travel
# caster through travel
# toe through travel (toe minus steer angle, if app)