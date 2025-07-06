# default
import time
from typing import Dict, Tuple

# third party
import numpy as np

# ours
from models.double_a_arm_numeric import DoubleAArmNumeric
from models.semi_trailing_numeric import SemiTrailingLinkNumeric
from scripts.hardpoints import Vehicle
from scripts.plotter import DoubleAArmPlotter, SemiTrailingLinkPlotter, CharacteristicPlotter, PointPlotter, SCALAR_CHARACTERISTIC, POINT_AXIS
from scripts.utils.wheel_utils import wheel_attitude

if __name__ == "__main__":
    vehicle: Vehicle = Vehicle("hardpoints/2024_final.yml")

    ### -- UNCOMMENT FOR FRONT SUSPENSION SIM -- ###

    # hp = vehicle.front_hp
    # solver = DoubleAArmNumeric(hp)

    # # plotters
    # # 3d double a arm
    # daa_plotter = DoubleAArmPlotter(hp)
    
    # # 2d scalar characteristics
    # camber_plot = CharacteristicPlotter(SCALAR_CHARACTERISTIC.CAMBER)
    # caster_plot = CharacteristicPlotter(SCALAR_CHARACTERISTIC.CASTER)
    # toe_plot = CharacteristicPlotter(SCALAR_CHARACTERISTIC.TOE)

    # # 2d specific points
    # # ib_tr_plot   = PointPlotter("tie_rod_chassis", POINT_AXIS.Y)

    # sh_len = np.linspace(-200, 400, 300, True)
    # bp_len = np.linspace(-80, 120, 20, True)
    # st_len = np.linspace(-20, 20, 100, True)

    # for x in sh_len:
    #     step = solver.solve(travel_mm=x, steer_mm=0)
    #     if step:
    #         # 3d
    #         daa_plotter.update(step)

    #         # 2d
    #         att = wheel_attitude(step)
    #         camber_plot.update(att)
    #         caster_plot.update(att)
    #         toe_plot.update(att)

    #         # ib_tr_plot.update(step)

    # # for y in st_len:
    # #     step = solver.solve(travel_mm=0, steer_mm=y)
    # #     print(y)
    # #     if step:
    # #         # 3d
    # #         daa_plotter.update(step)

    # #         # 2d
    # #         att = wheel_attitude(step)
    # #         camber_plot.update(att)
    # #         caster_plot.update(att)
    # #         toe_plot.update(att)

    # #         # ib_tr_plot.update(step)

    # solver.reset()

    # daa_plotter.display()
    # camber_plot.display()
    # caster_plot.display()
    # toe_plot.display()
    # # ib_tr_plot.display()

    ### -- UNCOMMENT FOR REAR SUSPENSION SIM -- ###

    # hp = vehicle.rear_hp
    # solver = SemiTrailingLinkNumeric(hp)

    # # plotters
    # # 3d semi trailing link
    # stl_plotter = SemiTrailingLinkPlotter(hp)

    # # 2d scalar characteristics
    # camber_plot = CharacteristicPlotter(SCALAR_CHARACTERISTIC.CAMBER)
    # toe_plot = CharacteristicPlotter(SCALAR_CHARACTERISTIC.TOE)

    # sh_len = np.linspace(-200, 400, 300, True)
    # for x in sh_len:
    #     step = solver.solve(travel_mm=x)
    #     if step:
    #         # 3d
    #         stl_plotter.update(step)

    #         # 2d
    #         att = wheel_attitude(step)
    #         camber_plot.update(att)
    #         toe_plot.update(att)

    # solver.reset()
    # stl_plotter.display()
    # camber_plot.display()
    # toe_plot.display()