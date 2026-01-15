# default
import time
import copy
import random
from multiprocessing import Pool, cpu_count
from typing import Dict

# third-party
import yaml
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm

# ours
from models.hardpoints import Vehicle
from scripts.simulation import WheelAttitudeSimulation
from scripts.utils.wheel_utils import wheel_attitude

class SuspensionOptimizer:
    def __init__(self, config: Dict, corner_map: Dict):
        self.config = config
        self.points_to_opt = config["OPTIMIZE_POINTS"]
        self.xyz_limits = config["BOUNDING_BOX_XYZ_LIMITS"]
        self.resolution = config["GRID_SEARCH_RESOLUTION"]
        self.corner_map = corner_map
        self.iter_count = 0 
        self.silent = False
        
        # History objects
        self.current_history = []   
        self.all_trajectories = [] 
        self.map_data = {} # Stores the random sampling of the search space
        
        # Load Base Data
        with open(config["HARDPOINT_FILE"], 'r') as f:
            self.base_hardpoints = yaml.safe_load(f)
            
        with open(config["SIM_CONFIG_FILE"], 'r') as f:
            self.sim_config = yaml.safe_load(f)
        self.sim_config["SIMULATION"] = "travel"

        # Setup initial vector x0, bounds, and identify free indices
        self.x0 = []
        self.bounds = []
        self.free_indices = [] # Indices of variables that actually move
        
        vehicle_name = list(self.base_hardpoints.keys())[0] 
        self.hp_subset = self.base_hardpoints[vehicle_name][config["HALF"]]

        # Flatten all points into x0 and identify bounds
        global_idx = 0
        for pt in self.points_to_opt:
            coords = self.hp_subset[pt]
            self.x0.extend(coords) 
            
            if pt not in self.xyz_limits:
                raise KeyError(f"Point '{pt}' missing from BOUNDING_BOX_XYZ_LIMITS.")
            
            point_limits = self.xyz_limits[pt]

            for i in range(3): # x, y, z
                original_val = coords[i]
                min_delta = point_limits[i][0] 
                max_delta = point_limits[i][1] 
                
                # Check if this degree of freedom is actually free
                if min_delta != max_delta:
                    self.free_indices.append(global_idx)

                self.bounds.append((original_val + min_delta, original_val + max_delta))
                global_idx += 1

    def _update_vehicle_data(self, current_coords: np.ndarray) -> Dict:
        data_copy = copy.deepcopy(self.base_hardpoints)
        vehicle_name = list(data_copy.keys())[0]
        target_half = data_copy[vehicle_name][self.config["HALF"]]
        
        idx = 0
        for pt in self.points_to_opt:
            target_half[pt] = current_coords[idx : idx+3].tolist()
            idx += 3
        return data_copy

    def objective_function(self, x):
        x_arr = np.array(x)
        updated_data = self._update_vehicle_data(x_arr)
        vehicle = Vehicle(updated_data)

        corner_attr = self.corner_map[(self.config["SIDE"], self.config["HALF"])]
        corner = getattr(vehicle, corner_attr)
        
        sim = WheelAttitudeSimulation(vehicle, self.sim_config)

        try:
            steps = sim.run(corner=corner)
        except Exception:
            return 1e6

        toes = [wheel_attitude(step)['toe'] for step in steps]
        toes = np.array(toes)
        
        if len(toes) == 0: return 1e6

        max_abs_toe = np.max(np.abs(toes))
        toe_range = np.max(toes) - np.min(toes)
        total_cost = max_abs_toe + toe_range

        if not self.silent:
            self.iter_count += 1
            # We store the full vector x plus the cost
            record = list(x)
            record.append(total_cost)
            self.current_history.append(record)

        return total_cost

    def map_search_space(self, num_samples=2000) -> None:
        print(f"\n--- Grid Calculating Search Space ({num_samples} samples) ---")
        
        inputs = []

        # Generate Random Inputs within bounds
        for _ in range(num_samples):
            candidate = []
            for bound in self.bounds:
                candidate.append(random.uniform(bound[0], bound[1]))
            inputs.append(candidate)

        self.silent = True
        total = len(inputs)
        cores = cpu_count()
        print(f"Evaluating {total} points using {cores} cores...")
        
        t0 = time.time()
        
        results = []
        with Pool(processes=cores) as pool:
            for i, res in enumerate(pool.imap(self.objective_function, inputs)):
                results.append(res)
                print(f"Grid Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
        
        print(f"Grid complete in {time.time() - t0:.2f}s.")
        
        # Store data for plotting
        # Convert inputs to numpy array for easy slicing
        input_arr = np.array(inputs)
        self.map_data = {
            'X': input_arr[:, 0], # Assumes Tie Rod Inboard X is index 0
            'Y': input_arr[:, 1], # Assumes Tie Rod Inboard Y is index 1
            'Z': input_arr[:, 2], # Assumes Tie Rod Inboard Z is index 2
            'Cost': np.array(results)
        }
        self.silent = False 

    def plot_results(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot cost cloud
        if self.map_data:
            sc = ax.scatter(
                self.map_data['X'], 
                self.map_data['Y'], 
                self.map_data['Z'], 
                c=self.map_data['Cost'], 
                cmap='jet', 
                alpha=0.5,
                s=20,
                marker='o',
                depthshade=False
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('Cost (Blue=Low, Red=High)')

        # Plot trajectories
        print(f"Plotting {len(self.all_trajectories)} trajectories...")
        
        # Make trajectory lines distinct (black/dark grey to stand out against cloud)
        for i, history in enumerate(self.all_trajectories):
            if not history: continue
            h_arr = np.array(history)
            
            # Plot the path
            ax.plot(h_arr[:, 0], h_arr[:, 1], h_arr[:, 2], color='black', linewidth=1, alpha=0.6)
            
            # Start point (Green)
            ax.scatter(h_arr[0, 0], h_arr[0, 1], h_arr[0, 2], color='green', marker='o', s=30, alpha=1.0)
            
            # End point (Red Star)
            ax.scatter(h_arr[-1, 0], h_arr[-1, 1], h_arr[-1, 2], color='red', marker='*', s=100, edgecolors='white', alpha=1.0)

        ax.set_xlabel('Inboard X (mm)')
        ax.set_ylabel('Inboard Y (mm)')
        ax.set_zlabel('Inboard Z (mm)')
        ax.set_title(f'Suspension Optimization Landscape\n')
        
        # Set initial view angle for better 3D perception
        ax.view_init(elev=20, azim=-45)
        plt.show()

    def generate_random_start(self) -> np.ndarray:
        return np.array([random.uniform(b[0], b[1]) for b in self.bounds])

    def run(self):
        print(f"--- Starting Multi-Start Optimization ---")
        
        # Map the cloud
        # Using user config resolution to determine sample count (res^2 or res^3 roughly)
        sample_count = self.resolution * self.resolution 
        self.map_search_space(num_samples=sample_count)
        
        num_starts = self.config['NUM_STARTS']
        print(f"\n--- Starting {num_starts} Random Optimization Runs ---")
        
        best_cost = float('inf')
        best_x = None

        for i in range(num_starts):
            x_start = self.generate_random_start()
            self.current_history = []
            
            # Use callback to update progress per step if needed, or just end of run
            res = minimize(
                self.objective_function, x_start, 
                method=self.config["METHOD"], bounds=self.bounds, 
                tol=self.config["TOLERANCE"], options={'maxiter': self.config["MAX_ITER"], 'disp': False}
            )
            
            self.all_trajectories.append(copy.deepcopy(self.current_history))
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
                
            print(f"Run {i+1}/{num_starts} ({(i+1)/num_starts*100:.1f}%) | Best: {best_cost:.4f}", end="\r")

        print(f"\n\nOptimization Complete. Best Global Cost: {best_cost:.4f}")
        
        # --- PRINT RESULTS TABLE ---
        print(f"{'Rank':<5} | {'Cost':<10} | {'Inb X':<9} | {'Inb Y':<9} | {'Inb Z':<9} | {'Del X':<8} | {'Del Y':<8} | {'Del Z':<8}")
        print("-" * 90)

        # Capture originals for Delta calc
        orig_x, orig_y, orig_z = self.x0[0], self.x0[1], self.x0[2]

        for i, traj in enumerate(self.all_trajectories):
            if not traj: continue
            
            # Sort this run's history by cost (last element in the list record)
            sorted_traj = sorted(traj, key=lambda rec: rec[-1])
            rank = i
            p = sorted_traj[0]  # Best point in this trajectory
            curr_x, curr_y, curr_z = p[0], p[1], p[2]
            
            # p is the full vector x + cost at end
            cost = p[-1]            
            dx = curr_x - orig_x
            dy = curr_y - orig_y
            dz = curr_z - orig_z
            
            print(f"{rank+1:<5} | {cost:<10.4f} | {curr_x:<9.3f} | {curr_y:<9.3f} | {curr_z:<9.3f} | {dx:<+8.3f} | {dy:<+8.3f} | {dz:<+8.3f}")

        idx = 0
        for pt in self.points_to_opt:
            original = self.x0[idx:idx+3]
            new = best_x[idx:idx+3]
            print(f"\nGlobal Best {pt}:")
            print(f"  Old:   [{original[0]:.3f}, {original[1]:.3f}, {original[2]:.3f}]")
            print(f"  New:   [{new[0]:.3f}, {new[1]:.3f}, {new[2]:.3f}]")
            diff = new - np.array(original)
            print(f"  Delta: [{diff[0]:+.3f}, {diff[1]:+.3f}, {diff[2]:+.3f}]")
            idx += 3

        self.plot_results()
        return best_x

def run_optimizer(corner_map, sim_config_path="sim_config.yml", opt_config_path="opt_config.yml"):
    with open(sim_config_path, "r") as f:
        sim_config = yaml.safe_load(f)
    with open(opt_config_path, "r") as f:
        opt_config = yaml.safe_load(f)

    xyz_limits_dict = {}
    optimize_points_list = []
    
    for point_name, ranges in opt_config["FREE_POINTS"].items():
        optimize_points_list.append(point_name)
        xyz_limits_dict[point_name] = [ranges['x'], ranges['y'], ranges['z']]

    final_config = {
        "HARDPOINT_FILE": f"hardpoints/{sim_config['HARDPOINTS']}.yml",
        "SIM_CONFIG_FILE": sim_config_path,
        "SIDE": sim_config["SIDE"],
        "HALF": sim_config["HALF"],
        "OPTIMIZE_POINTS": optimize_points_list,
        "BOUNDING_BOX_XYZ_LIMITS": xyz_limits_dict,
        "GRID_SEARCH_RESOLUTION": int(opt_config["GRID_SEARCH_RESOLUTION"]),
        "METHOD": opt_config["METHOD"],
        "TOLERANCE": float(opt_config["TOLERANCE"]),
        "MAX_ITER": int(opt_config["MAX_ITER"]),
        "NUM_STARTS": int(opt_config["NUM_STARTS"])
    }

    opt = SuspensionOptimizer(final_config, corner_map)
    opt.run()