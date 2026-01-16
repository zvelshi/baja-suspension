# default
from typing import List, Dict, Tuple

# third-party
import matplotlib.pyplot as plt
import numpy as np

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from models.vehicle import Vehicle
from utils.geometry import get_wheel_attitude

def _compute_wheel_cylinder(wc, step, radius, width, n_theta=40):
    """
    Computes the visual mesh for the wheel cylinder.
    """
    wc = np.asarray(wc, dtype=float)
    axis = None

    # Try to get axis from step
    if "wheel_axis" in step:
        a = np.asarray(step["wheel_axis"], dtype=float)
        if np.linalg.norm(a) > 1e-9:
            axis = a / np.linalg.norm(a)

    # Default to Global Y
    if axis is None:
        axis = np.array([0.0, 1.0, 0.0])

    # Basis vectors
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    
    u = tmp - np.dot(tmp, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    c_front = wc - 0.5 * width * axis
    c_back  = wc + 0.5 * width * axis

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cf = c_front[:, None] + radius * (u[:, None] * cos_t + v[:, None] * sin_t)
    cb = c_back[:, None]  + radius * (u[:, None] * cos_t + v[:, None] * sin_t)
    return cf, cb

class Plotter:
    def __init__(self, title="Simulation Results"):
        self.title = title
        self.figures = []

    def show(self):
        plt.show()

    def _set_axes_equal(self, ax):
        """Forces 3D axes to be equal scale."""
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def plot_3d_corner(self, steps: List[Dict], hp):
        """
        Plots a single corner (Double A-Arm or Semi-Trailing Link).
        Requires the Hardpoints object (hp) to draw static chassis points.
        """
        if not steps: return
        step = steps[-1] # Plot the last state

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{self.title} - 3D Geometry")

        if isinstance(hp, DoubleAArm):
            self._render_double_a_arm(ax, step, hp)
        elif isinstance(hp, SemiTrailingLink):
            self._render_semi_trailing(ax, step, hp)
        
        self._set_axes_equal(ax)
        self.figures.append(fig)

    def plot_3d_ackermann(self, steps: List[Dict], vehicle: Vehicle):
        """
        Plots the full front end (Left and Right Double A-Arms).
        """
        if not steps: return
        step = steps[-1]

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Ackermann Geometry (Full Front)")

        # Left Corner
        if 'left' in step:
            self._render_double_a_arm(ax, step['left'], vehicle.front_left.hardpoints, color_main='blue')
        
        # Right Corner
        if 'right' in step:
            self._render_double_a_arm(ax, step['right'], vehicle.front_right.hardpoints, color_main='red')

        self._set_axes_equal(ax)
        self.figures.append(fig)

    def _render_double_a_arm(self, ax, step, hp: DoubleAArm, color_main='black'):
        # Helper to plot line
        def line(p1, p2, c=color_main, lw=1.5):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, lw=lw)

        # A-Arms (Chassis -> Upright)
        line(hp.uf, step['ubj']) # Upper Front
        line(hp.ur, step['ubj']) # Upper Rear
        line(hp.lf, step['lbj']) # Lower Front
        line(hp.lr, step['lbj']) # Lower Rear

        # Upright
        line(step['lbj'], step['ubj'], c=color_main, lw=2)

        # Steering (Tie Rod)
        line(step['tr_ib'], step['tr_ob'], c='green', lw=2)

        # Shock
        line(hp.s_ib, step['s_ob'], c='black', lw=2)

        # Axle
        if 'piv_ob' in step:
            line(hp.piv_ib, step['piv_ob'], c='red', lw=2)
            line(step['piv_ob'], step['wc'], c='red', lw=2)

        # Wheel
        try:
            cf, cb = _compute_wheel_cylinder(step['wc'], step, hp.wr, hp.ww)
            ax.plot(cf[0], cf[1], cf[2], color='grey', alpha=0.3)
            ax.plot(cb[0], cb[1], cb[2], color='grey', alpha=0.3)
        except:
            pass

        # Points
        all_pts = [hp.uf, hp.ur, hp.lf, hp.lr, hp.s_ib, step['ubj'], step['lbj'], step['wc']]
        pts = np.array(all_pts)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', s=20)

    def _render_semi_trailing(self, ax, step, hp: SemiTrailingLink, color_main='black'):
        def line(p1, p2, c=color_main, lw=1.5):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, lw=lw)

        # Trailing Link (Front Pivot -> Camber Links Outboard)
        line(hp.tl_f, step['ucl_ob']) 
        line(hp.tl_f, step['lcl_ob'])

        # Camber Links
        line(hp.ucl_ib, step['ucl_ob'])
        line(hp.lcl_ib, step['lcl_ob'])

        # Shock
        line(hp.s_ib, step['s_ob'])

        # Axle
        if 'piv_ob' in step:
            line(hp.piv_ib, step['piv_ob'], c='red', lw=2) # Axle Shaft
            line(step['piv_ob'], step['wc'], c='red', lw=2) # Stub Axle

        # Wheel
        try:
            cf, cb = _compute_wheel_cylinder(step['wc'], step, hp.wr, hp.ww)
            ax.plot(cf[0], cf[1], cf[2], color='grey', alpha=0.3)
            ax.plot(cb[0], cb[1], cb[2], color='grey', alpha=0.3)
        except: pass
        
        # Points
        pts = [hp.tl_f, hp.ucl_ib, hp.lcl_ib, hp.s_ib, step['wc']]
        pts = np.array(pts)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', s=20)

    def plot_kinematics_curves(self, steps: List[Dict]):
        """
        Plots the 5 curves: Camber, Caster, Toe, Plunge, Axle Angles.
        """
        if not steps: return

        # Prepare Data
        attitudes = [get_wheel_attitude(s) for s in steps]
        
        # Extract Axle Data
        plunge = []
        angle_ib = []
        angle_ob = []
        for s in steps:
            if 'axle_data' in s:
                plunge.append(s['axle_data'].get('plunge_mm', 0))
                angle_ib.append(s['axle_data'].get('angle_ib_deg', 0))
                angle_ob.append(s['axle_data'].get('angle_ob_deg', 0))
            else:
                plunge.append(0); angle_ib.append(0); angle_ob.append(0)

        # X-Axis Definition
        if 'travel_mm' in steps[0]:
            xs = [s.get('travel_mm', 0) for s in steps]
            xlabel = "Shock Travel [mm]"
        elif 'steer_mm' in steps[0]:
            xs = [s.get('steer_mm', 0) for s in steps]
            xlabel = "Rack Travel [mm]"
        else:
            xs = range(len(steps))
            xlabel = "Step"

        # Setup Figure (2 Rows, 3 Cols)
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"{self.title} - Kinematics Dashboard")
        
        # Row 1: Wheel Attitude
        ax_camber = fig.add_subplot(2, 3, 1)
        ax_caster = fig.add_subplot(2, 3, 2)
        ax_toe    = fig.add_subplot(2, 3, 3)
        
        # Row 2: Axle Stats
        ax_plunge = fig.add_subplot(2, 3, 4)
        ax_angles = fig.add_subplot(2, 3, 5)

        # Plot Camber
        ax_camber.plot(xs, [a['camber'] for a in attitudes], '-o', markersize=3)
        ax_camber.set_title("Camber [deg]")
        ax_camber.set_xlabel(xlabel)
        ax_camber.grid(True)

        # Plot Caster
        ax_caster.plot(xs, [a['caster'] for a in attitudes], '-o', markersize=3)
        ax_caster.set_title("Caster [deg]")
        ax_caster.set_xlabel(xlabel)
        ax_caster.grid(True)

        # Plot Toe
        ax_toe.plot(xs, [a['toe'] for a in attitudes], '-o', markersize=3)
        ax_toe.set_title("Toe [deg]")
        ax_toe.set_xlabel(xlabel)
        ax_toe.grid(True)

        # Plot Plunge
        ax_plunge.plot(xs, plunge, '-o', color='orange', markersize=3)
        ax_plunge.set_title("Axle Plunge [mm]")
        ax_plunge.set_xlabel(xlabel)
        ax_plunge.grid(True)

        # Plot Axle Angles (IB vs OB)
        ax_angles.plot(xs, angle_ib, label="Inboard", color='red')
        ax_angles.plot(xs, angle_ob, label="Outboard", color='blue', linestyle='--')
        ax_angles.set_title("CV Joint Angles [deg]")
        ax_angles.set_xlabel(xlabel)
        ax_angles.legend()
        ax_angles.grid(True)

        self.figures.append(fig)

    def plot_ackermann_curve(self, steps: List[Dict]):
        """Specialized plot for Ackermann % vs Steer"""
        if not steps: return
        
        xs = [s['input'] for s in steps]
        ys = [s['ackermann_pct'] for s in steps]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(xs, ys, color='purple', lw=2)
        ax.set_title("Ackermann Percentage")
        ax.set_xlabel("Steering Rack [mm]")
        ax.set_ylabel("Ackermann [%]")
        ax.grid(True)
        self.figures.append(fig)

class CostCloudPlotter(Plotter):
    """
    Visualizes the optimization landscape.
    Plots every evaluated point in 3D space, colored by its cost.
    """
    def plot_cloud(self, evaluated_points: List[Tuple[float, np.ndarray]]):
        """
        evaluated_points: List of (cost, x_vector) tuples.
        """
        if not evaluated_points: return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{self.title} - Optimization Cost Landscape")
        
        costs = []
        xs, ys, zs = [], [], []
        
        # Unpack
        for cost, vec in evaluated_points:
            costs.append(cost)
            # Assume 3 vars for 3D plot. If <3 or >3, we handle logic
            if len(vec) >= 3:
                xs.append(vec[0])
                ys.append(vec[1])
                zs.append(vec[2])
            elif len(vec) == 2:
                xs.append(vec[0])
                ys.append(vec[1])
                zs.append(0)
            elif len(vec) == 1:
                xs.append(vec[0])
                ys.append(0)
                zs.append(0)
            
        costs = np.array(costs)
        
        # Filter out exact penalties
        valid_indices = costs < 900000 
        
        if np.any(valid_indices):
            valid_costs = costs[valid_indices]
            v_min = np.min(valid_costs)
            v_max = np.max(valid_costs)
        else:
            # Everything failed? Just show raw
            v_min, v_max = np.min(costs), np.max(costs)
            
        p = ax.scatter(xs, ys, zs, c=costs, cmap='jet', vmin=v_min, vmax=v_max, s=30, alpha=0.4)
        
        ax.set_xlabel("Var 1 (X)")
        ax.set_ylabel("Var 2 (Y)")
        ax.set_zlabel("Var 3 (Z)")
        
        cbar = fig.colorbar(p, ax=ax, label="Cost Function (Blue=Low, Red=High)")
        self.figures.append(fig)