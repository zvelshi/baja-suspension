# default
import random
import time
import itertools
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, cpu_count

# third-party
import numpy as np
from scipy.optimize import minimize

# ours
from models.vehicle import Vehicle
from optimization.objectives import OptimizationObjective
from simulations.scenarios import SuspensionSweep, AckermannScenario

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

    def objective_function(self, x):
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
                    cost = 1e6 # Penalty for failing to solve
                else:
                    cost = obj.calculate_cost(results)
                total_cost += cost
            except Exception:
                total_cost += 1e6

        return total_cost

    def _grid_search(self, resolution: int) -> List[Tuple[float, np.ndarray]]:
        """
        Generates a dense grid of points and evaluates them in PARALLEL.
        Returns sorted list of (cost, candidate_array).
        """

        num_vars = len(self.bounds)
        total_points = resolution ** num_vars

        print(f"-> Generating Grid: {resolution} steps per var ^ {num_vars} vars = {total_points} total points")
        if total_points > 100000:
            print("   WARNING: Large grid size. This may take a while.")

        axis_ranges = []
        for (lower, upper) in self.bounds:
            axis_ranges.append(np.linspace(lower, upper, resolution))

        grid_points = itertools.product(*axis_ranges)
        candidates = [np.array(pt) for pt in grid_points]

        evaluated = []
        t0 = time.time()

        cores = cpu_count()
        print(f"   Spinning up {cores} workers for grid search...")

        with Pool(processes=cores) as pool:
            results_iter = pool.imap(self.objective_function, candidates, chunksize=50)
            costs = []
            for i, cost in enumerate(results_iter):
                costs.append(cost)
                print(f"   Grid Progress: {i+1}/{total_points} ({(i+1)/total_points*100:.1f}%)", end='\r')
        
        print(f"\n   Grid Search complete in {time.time() - t0:.2f}s")

        for cost, cand in zip(costs, candidates):
            evaluated.append((cost, cand))
        
        self.history = evaluated
        evaluated.sort(key=lambda x: x[0])
        return evaluated

    def run(self):
        """
        Main optimization routine.
        """

        print(f"--- Starting Optimization ---")
        num_vars = len(self.x0)
        print(f"Optimizing {num_vars} variables for {self.target_id}")

        print("Checking initial point...")
        initial_cost = self.objective_function(np.array(self.x0, dtype=float))
        if initial_cost >= 1e6:
            print("FATAL: Initial point invalid (Geometry broken or constraints exceeded).")
            return np.array(self.x0)

        # GRID SEARCH
        grid_res = int(self.config.get("GRID_SEARCH_RESOLUTION", 10)) 
        sorted_results = self._grid_search(grid_res)
        
        # FINE POINT SEARCH AGENTS
        num_agents = int(self.config.get("FINE_POINT_SEARCH_AGENTS", 5))
        
        # Select Top N Candidates from Grid
        start_points = [res[1] for res in sorted_results[:num_agents]]

        # Ensure Original Design (x0) is checked
        x0_arr = np.array(self.x0, dtype=float)
        is_duplicate = False
        for p in start_points:
            if np.linalg.norm(p - x0_arr) < 1e-6:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Insert original at the front
            start_points.insert(0, x0_arr)

        print(f"\n-> Launching {len(start_points)} Fine-Tuning Agents (Top Grid Results + Original)...")
        
        best_cost = float('inf')
        best_x = None
        
        for i, x_start in enumerate(start_points):
            start_cost = self.objective_function(x_start)
            print(f"   Agent {i+1}: Start Cost {start_cost:.4f} ... ", end="")
            
            # Local Optimization (Gradient Descent)
            res = minimize(
                self.objective_function, 
                x_start, 
                method='L-BFGS-B', 
                bounds=self.bounds,
                tol=1e-4,
                options={'maxiter': 500}
            )
            
            print(f"Converged to {res.fun:.4f} with point {res.x}")
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x

        print(f"\nOptimization Complete.")
        print(f"Global Best Cost: {best_cost:.4f}")
        
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