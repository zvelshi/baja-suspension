import yaml
from typing import List

# ours
from models.hardpoints import Vehicle, DoubleAArm, SemiTrailingLink
from scripts.utils.plotter import (
    PlotterBase,
    DoubleAArmPlotter,
    SemiTrailingLinkPlotter,
    CharacteristicPlotter,
    AxleCharacteristicsPlotter,
    Ackermann3DPlotter,
    AckermannCurvePlotter,
    AXLE_CHARACTERISTIC,
    SCALAR_CHARACTERISTIC,
)
from scripts.simulation import (
    Simulation, 
    WheelAttitudeSimulation, 
    AckermannSimulation,
)
from scripts.utils.wheel_utils import (
    wheel_attitude,
    ackermann_percent,
)

def run_single_simulation(corner_map, sim_config_path: str = "sim_config.yml"):
    print(f"--- Running Single Simulation [Config: {sim_config_path}] ---")
    
    with open(sim_config_path, "r") as file:
        config = yaml.safe_load(file)

    hp_file = f"hardpoints/{config['HARDPOINTS']}.yml"
    with open(hp_file, 'r') as file:
        data = yaml.safe_load(file)

    vehicle = Vehicle(data)
    plots: List[PlotterBase] = []

    if config['SIMULATION'] == 'ackermann':
        print("-> Mode: Ackermann (Dual Front Corner)")
        simulation = AckermannSimulation(vehicle, config)
        steps = simulation.run()

        # Find the front corners
        fl = vehicle.front_left
        fr = vehicle.front_right

        # Init plotters
        if config["PLOTS"].get("3D", False):
            # We initialize with the hardpoint objects
            plots.append(Ackermann3DPlotter(fl.hardpoints, fr.hardpoints))
        
        if config["PLOTS"].get("TOE", False):
            plots.append(AckermannCurvePlotter())

        # Update Loop
        for step in steps: 
            steer_mm = step["input"]["steer_mm"]
            pct = ackermann_percent(step, vehicle, steer_mm)
            
            for plot in plots:
                if isinstance(plot, Ackermann3DPlotter):
                    plot.update(step)
                elif isinstance(plot, AckermannCurvePlotter):
                    plot.update(step, pct)

    else:
        # Which corner are we simulating?
        side = config.get("SIDE", "right")
        half = config.get("HALF", "front")
        corner = getattr(vehicle, corner_map[(side, half)])

        print(f"-> Mode: Single Corner ({half}-{side})")

        # Pick simulation
        simulation: Simulation
        if config['SIMULATION'] == 'steer' or config['SIMULATION'] == 'travel' or config['SIMULATION'] == 'steer_travel':
            simulation = WheelAttitudeSimulation(corner, config)
            steps = simulation.run(corner)
        else:
            raise ValueError(f"Unknown simulation: {config['SIMULATION']}")

        # Create and populate plots
        if config["PLOTS"]["CAMBER"]:
            plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.CAMBER))
        if config["PLOTS"]["CASTER"] and half == 'front':
            plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.CASTER))
        if config["PLOTS"]["TOE"]:
            plots.append(CharacteristicPlotter(SCALAR_CHARACTERISTIC.TOE))

        if config["PLOTS"].get("AXLE_PLUNGE", False):
            plots.append(AxleCharacteristicsPlotter(AXLE_CHARACTERISTIC.PLUNGE))
        if config["PLOTS"].get("AXLE_ANGLES", False):
            plots.append(AxleCharacteristicsPlotter(AXLE_CHARACTERISTIC.ANGLE_IB))
            plots.append(AxleCharacteristicsPlotter(AXLE_CHARACTERISTIC.ANGLE_OB))

        if config["PLOTS"].get("3D", False):
            hp = corner.hardpoints
            if half == 'front' and isinstance(hp, DoubleAArm):
                plots.append(DoubleAArmPlotter(hp))
            elif half == 'rear' and isinstance(hp, SemiTrailingLink):
                plots.append(SemiTrailingLinkPlotter(hp))

        # Update 2D plots across steps
        if steps:
            for st in steps:
                att = wheel_attitude(st)
                
                for plot in plots:
                    if isinstance(plot, CharacteristicPlotter):
                        plot.update(att)
                    elif isinstance(plot, AxleCharacteristicsPlotter):
                        plot.update(st)
                    
            if config["PLOTS"]["3D"] and plots:
                if isinstance(plots[-1], (DoubleAArmPlotter, SemiTrailingLinkPlotter)):
                    plots[-1].update(steps[-1])

    # Display all plots
    print("-> Displaying Plots...")
    for plot in plots:
        plot.display()