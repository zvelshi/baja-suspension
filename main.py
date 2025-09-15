# default
import time
from typing import Dict, Tuple, List
import yaml

# third party
import numpy as np

# ours
from models.double_a_arm_numeric import DoubleAArmNumeric
from models.semi_trailing_numeric import SemiTrailingLinkNumeric
from scripts.hardpoints import Vehicle, DoubleAArm, SemiTrailingLink
from scripts.plotter import PlotterBase, DoubleAArmPlotter, SemiTrailingLinkPlotter, CharacteristicPlotter, SCALAR_CHARACTERISTIC
from scripts.utils.wheel_utils import wheel_attitude

if __name__ == "__main__":
    with open("sim_config.yml", "r") as file:
        config = yaml.safe_load(file)

    hp_file = f"hardpoints/{config['HARDPOINTS']}.yml"

    vehicle: Vehicle = Vehicle(hp_file)

    plots: List[PlotterBase] = []
    if config["PLOTS"]["CAMBER"]:
        plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.CAMBER))

    if config["PLOTS"]["CASTER"] and config['TYPE'] == 'front':
        plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.CASTER))

    if config["PLOTS"]["TOE"]:
        plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.TOE))

    if config['TYPE'] == 'front':

        hp = vehicle.front_hp
        if not isinstance(hp, DoubleAArm):
            raise TypeError("Expected DoubleAArm for front suspension, got {type(hp).__name__}")
        solver = DoubleAArmNumeric(hp)

        if config["PLOTS"]["3D"]:
            plots.append(DoubleAArmPlotter(hp))

        travel_values = (
            np.linspace(config['TRAVEL']['MIN'], config['TRAVEL']['MAX'], config['SIM_STEPS'], True)
            if config['TRAVEL']['ENABLE'] else [config['TRAVEL']['VALUE']]
        )

        steer_values = (
            np.linspace(config['STEER']['MIN'], config['STEER']['MAX'], config['SIM_STEPS'], True)
            if config['STEER']['ENABLE'] else [config['STEER']['VALUE']]
        )

        for travel, steer in zip(travel_values, steer_values):
            step = solver.solve(bump_z=travel, steer_mm=steer)

            if step:

                # 3d
                plots[-1].update(step)

                # 2d
                att = wheel_attitude(step)
                for plot in plots[:-1]:
                    plot.update(att)

        solver.reset()

        for plot in plots:
            plot.display()

    elif config['TYPE'] == 'rear':

        hp = vehicle.rear_hp
        if not isinstance(hp, SemiTrailingLink):
            raise TypeError("Expected SemiTrailingLink for rear suspension, got {type(hp).__name__}")
        solver = SemiTrailingLinkNumeric(hp)

        if config["PLOTS"]["3D"]:
            plots.append(SemiTrailingLinkPlotter(hp))

        travel_values = (
            np.linspace(config['TRAVEL']['MIN'], config['TRAVEL']['MAX'], config['SIM_STEPS'], True)
            if config['TRAVEL']['ENABLE'] else [config['TRAVEL']['VALUE']]
        )

        for travel in travel_values:
            step = solver.solve(travel_mm=travel)
            if step:
                # 3d
                plots[-1].update(step)

                # 2d
                att = wheel_attitude(step)
                for plot in plots[:-1]:
                    plot.update(att)

        solver.reset()
        for plot in plots:
            plot.display()
    else:
        raise Exception(f"Invalid sim_type; {config['TYPE']}")