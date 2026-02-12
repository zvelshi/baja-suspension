# default
import time
from typing import List, Dict

# third-party
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ours
from models.vehicle import Vehicle
from optimization.objectives import OptimizationObjective
from simulations.scenarios import SuspensionSweep, AckermannScenario
from utils.misc import log_to_file

class SuspensionProblem(ElementwiseProblem):
    def __init__(self, optimizer):
        self.opt = optimizer
        super().__init__(
            n_var=len(optimizer.x0),
            n_obj=len(optimizer.objectives),
            xl=np.array([b[0] for b in optimizer.bounds]),
            xu=np.array([b[1] for b in optimizer.bounds])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x_str = ", ".join([f"{v:.4f}" for v in x])
        log_to_file(f"[EVAL] Testing Design: [{x_str}]")

        self.opt._apply_hardpoints(x)
        
        costs = []
        sim_cache = {}

        for obj in self.opt.objectives:
            s_type = obj.get_scenario_type()
            
            if s_type not in sim_cache:
                run_config = self.opt.config.copy()
                run_config["SIMULATION"] = s_type

                scenario_cls = self.opt._get_scenario_class(s_type)
                scenario = scenario_cls(self.opt.vehicle, run_config)
                
                try:
                    results = scenario.run()
                    sim_cache[s_type] = results
                except Exception as e:
                    log_to_file(f"  [CRASH] Sim '{s_type}' failed: {e}")
                    sim_cache[s_type] = None

            results = sim_cache[s_type]

            if not results:
                log_to_file(f"  [FAIL] {obj.name}: Invalid Geometry (Cost=1e6)")
                costs.append(1e6)
            else:
                try:
                    val = obj.calculate_cost(results)
                    costs.append(val)
                except Exception as e:
                    log_to_file(f"  [ERROR] {obj.name} cost calc failed: {e}")
                    costs.append(1e6)

        out["F"] = np.array(costs)

        c_str = ", ".join([f"{c:.6f}" for c in costs])
        log_to_file(f"  -> Result Costs: [{c_str}]")

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

        self.target_id = [0, 0] # Front Left default
        if config.get("HALF") == 'rear':
            self.target_id[1] = 1 
        if config.get("SIDE") == 'right':
            self.target_id[0] = 1 

        self.target_corner = vehicle.get_corner_from_id(self.target_id)

        self.pareto_front = None
        self.pareto_set = None

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

def run(self):
        """
        Main optimization routine.
        """

        print(f"--- Starting MOO ---")

        num_vars = len(self.x0)
        num_objs = len(self.objectives)
        
        pop_size = self.config.get("POP_SIZE", 40)
        n_offsprings = self.config.get("N_OFFSPRINGS", 10)
        
        print(f"Optimizing {num_vars} variables for {num_objs} objectives.")
        print(f"Population: {pop_size} | Offspring/Gen: {n_offsprings}")
        log_to_file(f"Setup: Vars={num_vars}, Objs={num_objs}, Pop={pop_size}, Offspring={n_offsprings}")
        log_to_file(f"Bounds: {self.bounds}")

        problem = SuspensionProblem(self)
        xl = np.array([b[0] for b in self.bounds])
        xu = np.array([b[1] for b in self.bounds])
        initial_pop = np.random.random((pop_size, num_vars)) * (xu - xl) + xl
        if len(self.x0) > 0:
            initial_pop[0, :] = np.array(self.x0)
            log_to_file(f"Seeding Initial Design: {self.x0}")

        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=initial_pop,
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", self.config.get("MAX_GEN", 50))

        t0 = time.time()
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True
        )

        self.pareto_front = res.F
        self.pareto_set = res.X

        duration = time.time() - t0
        print(f"\nOptimization Complete in {duration:.2f}s.")
        print(f"Found {len(res.F)} non-dominated solutions (Pareto Front).")

        log_to_file("\n" + "="*50)
        log_to_file(f"OPTIMIZATION RESULTS (Time: {duration:.2f}s)")
        log_to_file(f"Pareto Front Size: {len(res.F)}")
        log_to_file("="*50)

        F_safe = res.F
        if F_safe.ndim == 1:
            F_safe = F_safe.reshape(-1, 1)

        for i, (costs, design) in enumerate(zip(F_safe, res.X)):
            c_str = ", ".join([f"{c:.6f}" for c in costs])
            d_str = ", ".join([f"{d:.4f}" for d in design])
            log_to_file(f"Solution {i:03d}: Costs=[{c_str}] | Design=[{d_str}]")
        log_to_file("="*50 + "\n")
        return res