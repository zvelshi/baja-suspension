# default
import time
from typing import List, Dict
from multiprocessing import Manager

# third-party
import numpy as np
from scipy.optimize import differential_evolution

# ours
from models.vehicle import Vehicle
from optimization.objectives import OptimizationObjective
from simulations.scenarios import SuspensionSweep, AckermannScenario
from utils.misc import parse_time

class SuspensionOptimizer:
    def __init__(
        self, 
        vehicle: Vehicle, 
        config: Dict, 
        objectives: List[OptimizationObjective]
    ):
        self.vehicle = vehicle
        self.config = config
        self.objectives = objectives
        
        self.bounds = []
        self.x0 = []
        self.points_map = [] 
        
        # Determine target corner from config for points parsing
        self.target_id = [0, 0] # Front Left default
        if config.get("HALF") == 'rear':
            self.target_id[1] = 1 
        if config.get("SIDE") == 'right':
            self.target_id[0] = 1 
            
        self.target_corner = vehicle.get_corner_from_id(self.target_id)
        self.history = []
        self._parse_config_bounds()

    def _parse_config_bounds(self):
        target_hp = self.target_corner.hardpoints
        
        if "FREE_POINTS" not in self.config:
            return

        for pt_name, axes_limits in self.config["FREE_POINTS"].items():
            current_xyz = getattr(target_hp, pt_name)
            axis_map = {'x': 0, 'y': 1, 'z': 2}

            for axis_char, axis_idx in axis_map.items():
                if axis_char in axes_limits:
                    limits = axes_limits[axis_char]
                    if limits[0] != limits[1]:
                        current_val = current_xyz[axis_idx]
                        self.x0.append(current_val)
                        
                        lower_bound = current_val + limits[0]
                        upper_bound = current_val + limits[1]
                        self.bounds.append((lower_bound, upper_bound))
                        
                        self.points_map.append((self.target_id, pt_name, axis_idx))

    def _apply_hardpoints(self, x: np.ndarray):
        """
        Modifies vehicle geometry in-place, both sides.
        """

        for val, (c_id, pt_name, axis_idx) in zip(x, self.points_map):
            corner = self.vehicle.get_corner_from_id(c_id)
            pt_array = getattr(corner.hardpoints, pt_name)
            pt_array[axis_idx] = val

            # Calculate Mirror ID: Flip side bit (0->1 or 1->0)
            side, axle = c_id
            mirror_id = [1 - side, axle]

            try:
                mirror_corner = self.vehicle.get_corner_from_id(mirror_id)
                mirror_pt_array = getattr(mirror_corner.hardpoints, pt_name)

                # Mirror L/R (about XZ plane):
                # X (0) and Z (2) are identical.
                # Y (1) is inverted (-val).
                if axis_idx == 1:
                    mirror_pt_array[axis_idx] = -val
                else:
                    mirror_pt_array[axis_idx] = val 
            except ValueError:
                pass

    def _get_scenario_class(self, key: str):
        if key in ['steer', 'travel', 'steer_travel']: 
            return SuspensionSweep

        if key == 'ackermann': 
            return AckermannScenario

        raise ValueError(f"Unknown scenario type: {key}")

    def objective_function(self, x, shared_history=None):
        """
        Total Cost = Sum(Objective_i_Cost)
        Runs multiple scenarios if objectives require different simulations.
        """

        self._apply_hardpoints(x)
        total_cost = 0.0
        
        for obj in self.objectives:
            run_config = self.config.copy()
            run_config["SIMULATION"] = obj.get_scenario_type()
            scenario_cls = self._get_scenario_class(obj.get_scenario_type())
            scenario = scenario_cls(self.vehicle, run_config)

            try:
                results = scenario.run()
                if not results:
                    cost = 1e6 
                else:
                    cost = obj.calculate_cost(results)
                total_cost += cost
            except Exception:
                total_cost += 1e6

        # Safely append to the multiprocessing list if it was passed in
        if shared_history is not None:
            shared_history.append((total_cost, x.copy()))

        return total_cost

    def run(self):
        """
        Main optimization routine.
        """

        print(f"--- Starting Optimization ---")
        num_vars = len(self.x0)
        print(f"Optimizing {num_vars} variables for {self.target_id}")

        max_generations = self.config.get("DE_MAX_GENERATIONS", 100)
        popsize = self.config.get("DE_POPULATION_SIZE", 15)
        tol = self.config.get("DE_TOLERANCE", 0.01)
        mutation = tuple(self.config.get("DE_MUTATION", [0.5, 1.0]))
        recombination = self.config.get("DE_RECOMBINATION", 0.7)
        workers = -1

        print("Checking initial point...")
        initial_cost = self.objective_function(np.array(self.x0, dtype=float))
        if initial_cost >= 1e6:
            print("FATAL: Initial point invalid (Geometry broken or constraints exceeded).")
            return np.array(self.x0)

        print(f"\n-> Launching Differential Evolution Global Optimizer...")
        
        t0 = time.time()
        self.current_gen = 0
        manager = Manager()
        shared_history = manager.list()

        def custom_progress_bar(xk, convergence):
            self.current_gen += 1
            elapsed = time.time() - t0
            rate = self.current_gen / elapsed if elapsed > 0 else 0
            remaining_gens = max_generations - self.current_gen
            etr_seconds = remaining_gens / rate if rate > 0 else 0
            
            conv_pct = min(convergence * 100, 100.0)
            etr_str = parse_time(etr_seconds)
            print(f"   Generation: {self.current_gen}/{max_generations} | Swarm Convergence: {conv_pct:.1f}% | ETR: {etr_str}   ", end='\r')

        # Run the optimizer
        res = differential_evolution(
            self.objective_function, 
            bounds=self.bounds,
            args=(shared_history,),
            strategy='best1bin',
            maxiter=max_generations,
            popsize=popsize,
            tol=tol, 
            mutation=mutation,
            recombination=recombination,
            disp=False,
            callback=custom_progress_bar,
            workers=workers,
            updating='deferred'
        )

        best_x = res.x
        best_cost = res.fun
        self.history = list(shared_history)

        print(f"\n\nOptimization Complete in {time.time() - t0:.2f}s.")
        print(f"Global Best Cost: {best_cost:.4f} (Total Geometry Evals: {res.nfev})")
        
        # Compare Best vs Original
        print("-" * 65)
        print(f"{'Variable':15} | {'Original':<10} | {'New':<10} | {'Delta':<10}")
        print("-" * 65)
        
        orig_vals = self.x0
        
        for i, val in enumerate(best_x):
            orig = orig_vals[i]
            delta = val - orig
            
            _, pt_name, axis = self.points_map[i]
            axis_char = ['X', 'Y', 'Z'][axis]
            label = f"{pt_name}.{axis_char}"
            
            print(f"{label:<15} | {orig:<10.3f} | {val:<10.3f} | {delta:<+10.3f}")

        return best_x