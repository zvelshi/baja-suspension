# default
import argparse
import sys

# third-party
import yaml

# ours
import optimization.objectives as opt_objs
from models.vehicle import Vehicle
from simulations.scenarios import SuspensionSweep, AckermannScenario
from optimization.engine import SuspensionOptimizer
from utils.plotting import Plotter, CostCloudPlotter

def load_vehicle(sim_config_path: str):
    """Helper to load config and vehicle objects."""
    with open(sim_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    hp_file = f"config/hardpoints/{config['HARDPOINTS']}.yml"
    with open(hp_file, 'r') as f:
        hp_data = yaml.safe_load(f)
        
    vehicle = Vehicle(hp_data)
    return vehicle, config

def handle_simulation(args):
    print(f"--- Simulation Mode: {args.config} ---")
    vehicle, config = load_vehicle(args.config)
    
    sim_type = config.get("SIMULATION", "travel")
    scenario = None
    
    if sim_type == "ackermann":
        print("-> Running Ackermann Steering Geometry Analysis...")
        scenario = AckermannScenario(vehicle, config)
    
    elif sim_type in ["steer", "travel", "steer_travel"]:
        corner_id = [0, 0]
        if config["HALF"] == 'rear':
            corner_id[1] = 1
        elif config["SIDE"] == 'right':
            corner_id[0] = 1

        print(f"-> Running Single Corner Sweep: {corner_id} [{sim_type}]...")
        scenario = SuspensionSweep(vehicle, config)

    else:
        print(f"Error: Unknown simulation type '{sim_type}'")
        return

    results = scenario.run()
    if not results:
        print("No valid solution steps returned. Check geometry constraints.")
        return

    print(f"-> Generating Plots ({len(results)} steps)...")
    plotter = Plotter(title=f"Sim: {sim_type}")

    if sim_type == "ackermann":
        plotter.plot_3d_ackermann(results, vehicle)
        plotter.plot_ackermann_curve(results)
    else:
        id = [0, 0]
        if config["HALF"] == 'rear':
            id = [id[0], 1]
        if config["SIDE"] == 'right':
            id = [1, id[1]]
        corner = vehicle.get_corner_from_id(id)
        plotter.plot_3d_corner(results, corner.hardpoints)
        plotter.plot_kinematics_curves(results)
    plotter.show()

def handle_optimization(args):
    print(f"--- Optimization Mode ---")
    vehicle, sim_config = load_vehicle(args.sim_config)
    
    with open(args.opt_config, "r") as f:
        opt_config = yaml.safe_load(f)
    config = {**sim_config, **opt_config}

    obj_names = config["OBJECTIVES"]
    objectives = []
    print(f"-> Loading Objectives: {obj_names}")
    
    for name in obj_names:
        try:
            obj_cls = getattr(opt_objs, name)
        except AttributeError:
            print(f"FATAL ERROR: Objective class '{name}' not found in 'optimization.objectives'.")
            print("Please check the spelling in opt_config.yml or the class definition.")
            return
        objectives.append(obj_cls())

    optimizer = SuspensionOptimizer(vehicle, config, objectives)
    best_coords = optimizer.run()
    
    print("\nOptimization Complete.")
    print(f"Best Coordinates: {best_coords}")

    print("-> Plotting Optimization Cloud...")
    history_data = list(optimizer.history) 
    plotter = CostCloudPlotter(title="Differential Evolution Search")
    plotter.plot_cloud(history_data, optimizer.points_map)
    plotter.show()

def main():
    parser = argparse.ArgumentParser(description="Suspension Simulation & Optimization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")
    
    sim_parser = subparsers.add_parser("sim", help="Run a kinematic simulation")
    sim_parser.add_argument("--config", type=str, default="config/sim_config.yml", help="Path to sim config")

    opt_parser = subparsers.add_parser("opt", help="Run the optimizer")
    opt_parser.add_argument("--sim_config", type=str, default="config/sim_config.yml", help="Base sim config")
    opt_parser.add_argument("--opt_config", type=str, default="config/opt_config.yml", help="Optimization config")

    args = parser.parse_args()

    if args.command == "sim":
        handle_simulation(args)
    elif args.command == "opt":
        handle_optimization(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()