# default
import argparse
import sys

# ours
from scripts.single_sim import run_single_simulation
from scripts.optimizer import run_optimizer

CORNER_MAP = {
    ("left", "front"):  "front_left",
    ("right", "front"): "front_right",
    ("left", "rear"):   "rear_left",
    ("right", "rear"):  "rear_right",
}

def main():
    parser = argparse.ArgumentParser(description="Suspension Simulation & Optimization Tool")
    
    # Subcommands: 'sim' or 'opt'
    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")
    
    # SIMULATION MODE
    sim_parser = subparsers.add_parser("sim", help="Run a single simulation sweep")
    sim_parser.add_argument("--config", type=str, default="sim_config.yml", help="Path to simulation config file")

    # OPTIMIZATION MODE
    opt_parser = subparsers.add_parser("opt", help="Run the bump steer optimizer")
    opt_parser.add_argument("--sim_config", type=str, default="sim_config.yml", help="Path to base simulation config")
    opt_parser.add_argument("--opt_config", type=str, default="opt_config.yml", help="Path to optimization config")

    args = parser.parse_args()

    if args.command == "sim":
        run_single_simulation(CORNER_MAP, args.config)
    elif args.command == "opt":
        run_optimizer(CORNER_MAP, args.sim_config, args.opt_config)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()