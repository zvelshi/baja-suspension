# default
import time
from typing import Dict, Tuple, List
import yaml

# third party
import numpy as np

# ours
from scripts.hardpoints import Vehicle, DoubleAArm, SemiTrailingLink
from scripts.plotter import (
    PlotterBase,
    DoubleAArmPlotter,
    SemiTrailingLinkPlotter,
    CharacteristicPlotter,
    SCALAR_CHARACTERISTIC,
)
from scripts.simulation import (
    Simulation, 
    WheelAttitudeSimulation, 
    JackingSimulation
)
from scripts.utils.wheel_utils import wheel_attitude

CORNER_MAP = {
    ("left", "front"):  "front_left",
    ("right", "front"): "front_right",
    ("left", "rear"):   "rear_left",
    ("right", "rear"):  "rear_right",
}

if __name__ == "__main__":
    with open("sim_config.yml", "r") as file:
        config = yaml.safe_load(file)

    hp_file = f"hardpoints/{config['HARDPOINTS']}.yml"
    with open(hp_file, 'r') as file:
        data = yaml.safe_load(file)

    vehicle: Vehicle = Vehicle(data)

    # Which corner are we simulating?
    side = config.get("SIDE", "right") # default to right
    half = config.get("HALF", "front") # default to front
    corner = getattr(vehicle, CORNER_MAP[(side, half)])

    # Pick simulation
    simulation: Simulation
    if config['SIMULATION'] == 'wheel_attitude':
        simulation = WheelAttitudeSimulation(vehicle, config)
        steps = simulation.run(corner=corner)
    elif config['SIMULATION'] == 'jacking':
        simulation = JackingSimulation(vehicle, config)
        steps = simulation.run(half=half)
    else:
        raise ValueError("SIMULATION must be 'wheel_attitude' or 'jacking'.")

    # Create and populate plots
    plots: List[PlotterBase] = []
    if config["PLOTS"]["CAMBER"]:
        plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.CAMBER))
    if config["PLOTS"]["CASTER"] and half == 'front':
        plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.CASTER))
    if config["PLOTS"]["TOE"]:
        plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.TOE))

    if config["PLOTS"]["3D"]:
        hp = corner.hardpoints
        if half == 'front' and isinstance(hp, DoubleAArm):
            plots.append(DoubleAArmPlotter(hp))
        elif half == 'rear' and isinstance(hp, SemiTrailingLink):
            plots.append(SemiTrailingLinkPlotter(hp))

    # Update 2D plots across steps
    if steps:
        for st in steps:
            att = wheel_attitude(st)
            for plot in plots[:-1] if config["PLOTS"]["3D"] else plots:
                plot.update(att)

        # Update 3D plot after all steps
        if config["PLOTS"]["3D"] and plots:
            plots[-1].update(steps[-1])

    # Display all plots
    for plot in plots:
        plot.display()