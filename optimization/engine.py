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
from simulations.solvers import SingleCornerSolver
from utils.geometry import get_wheel_attitude

class OptimizationObjective(ABC):
    @abstractmethod
    def calculate_cost(self, results: List[Any]) -> float:
        pass
    
    @abstractmethod
    def get_simulation_inputs(self) -> List[Dict[str, float]]:
        """Returns a list of inputs (travel_mm, steer_mm) to run the solver on."""
        pass

class BumpSteerObjective(OptimizationObjective):
    def __init__(self, travel_range: tuple[float, float], steps: int = 10):
        self.travel_vals = np.linspace(travel_range[0], travel_range[1], steps)

    def get_simulation_inputs(self):
        # We want to sweep travel with 0 steer
        return [{"travel_mm": t} for t in self.travel_vals]

    def calculate_cost(self, results):
        if not results: 
            return 1e6
        
        toes = []
        for res in results:
            att = get_wheel_attitude(res)
            toes.append(att['toe'])
            
        toes = np.array(toes)
        
        # Minimize max absolute toe and the total range of toe change
        return np.max(np.abs(toes)) + (np.max(toes) - np.min(toes))

class SuspensionOptimizer:
    def __init__(
        self, 
        vehicle: Vehicle, 
        config: Dict, 
        objective: OptimizationObjective
    ):
        self.vehicle = vehicle
        self.config = config
        self.objective = objective
        
        self.bounds = []
        self.x0 = []
        self.points_map = [] 
        
        self.solver = SingleCornerSolver(self.vehicle)

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
        """Modifies vehicle geometry in-place."""
        for val, (c_id, pt_name, axis_idx) in zip(x, self.points_map):
            corner = self.vehicle.get_corner_from_id(c_id)
            pt_array = getattr(corner.hardpoints, pt_name)
            pt_array[axis_idx] = val

    def objective_function(self, x):
        """
        The core cost function.
        Returns a float cost (lower is better).
        """
        self._apply_hardpoints(x)

        sim_inputs = self.objective.get_simulation_inputs()
        results = []
        
        for inp in sim_inputs:
            try:
                s_mm = inp.get('steer_mm', 0.0)
                t_mm = inp.get('travel_mm', 0.0)

                step = self.solver.solve(
                    self.target_id, 
                    steer_mm=s_mm, 
                    travel_mm=t_mm
                )
                if step:
                    results.append(step)
                else:
                    pass
            except Exception:
                pass
        
        if not results: return 1e6

        return self.objective.calculate_cost(results)

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
            
            print(f"Converged to {res.fun:.4f}")
            
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