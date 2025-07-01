# default
from typing import Dict, List
from enum import Enum, auto

# third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ours
from scripts.hardpoints import DoubleAArmHardpoints

class SCALAR_CHARACTERISTIC(Enum):
    CAMBER = auto()
    CASTER = auto()
    TOE    = auto()

class DoubleAArmPlotter:
    def __init__(self, hp: DoubleAArmHardpoints):
        self.hp = hp
        self._path: list[np.ndarray] = []
        self._init_3d_plot()
    
    def _init_3d_plot(self):
        hp = self.hp
        fig = plt.figure(figsize=(7,7))
        self.ax: Axes3D = fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_zlabel("Z [mm]")
        self.ax.set_title("Double‑A‑Arm Corner")
        self.ax.view_init(elev=20, azim=30)
        self.ax.grid(True, linestyle=":", linewidth=0.5)

        # static inboard a arm points
        self.ax.scatter(hp.uf[0], hp.uf[1], hp.uf[2], color='black', s=25)
        self.ax.scatter(hp.ur[0], hp.ur[1], hp.ur[2], color='black', s=25)
        self.ax.scatter(hp.lf[0], hp.lf[1], hp.lf[2], color='black', s=25)
        self.ax.scatter(hp.lr[0], hp.lr[1], hp.lr[2], color='black', s=25)

        # static inboard a arm lines
        for pt in (hp.uf, hp.ur, hp.lf, hp.lr):
            self.ax.scatter(pt[0], pt[1], pt[2], c="black", s=25)

        self.ax.scatter(hp.shock_chassis[0], hp.shock_chassis[1], hp.shock_chassis[2], c="black", s=25)

        # dynamic ball joint points
        self.mov_scatter = self.ax.scatter([], [], [], c="black", s=25)

        # dynamic a arm lines
        self.upper_lines = [self.ax.plot([], [], [], c="black", lw=1.5)[0] for _ in range(2)]
        self.lower_lines = [self.ax.plot([], [], [], c="black", lw=1.5)[0] for _ in range(2)]
        self.shock_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.tie_rod_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]

        # path line for wheel center
        self.path_line, = self.ax.plot([], [], [], c="red", lw=1.5)

        static = np.vstack([hp.uf, hp.ur, hp.lf, hp.lr, hp.tr_chassis, hp.shock_chassis, hp.shock_a_arm])
        mins, maxs = static.min(0), static.max(0)
        centre, half = (mins+maxs)/2, (maxs-mins)/2 * 2.5
        for set_lim, c, h in zip((self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim), centre, half):
            set_lim(c-h, c+h)

        plt.ion(); plt.show(block=False)

    def update(self, step: Dict[str, np.ndarray]):
        hp = self.hp
        bjl = step["lower_ball_joint"]
        bju = step["upper_ball_joint"]
        wc  = step["wheel_center"]
        sha = step["shock_a_arm"]
        tr_chassis = step["tie_rod_chassis"]
        tr_upright = step["tie_rod_upright"]

        pts = np.vstack([bjl, bju, wc, sha])
        self.mov_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        # upper front‑rear
        for line, chassis in zip(self.upper_lines, (hp.uf, hp.ur)):
            line.set_data([chassis[0], bju[0]], [chassis[1], bju[1]])
            line.set_3d_properties([chassis[2], bju[2]])

        # lower front‑rear
        for line, chassis in zip(self.lower_lines, (hp.lf, hp.lr)):
            line.set_data([chassis[0], bjl[0]], [chassis[1], bjl[1]])
            line.set_3d_properties([chassis[2], bjl[2]])

        # shock
        self.shock_line.set_data([hp.shock_chassis[0], sha[0]], [hp.shock_chassis[1], sha[1]])
        self.shock_line.set_3d_properties([hp.shock_chassis[2], sha[2]])

        # tie rod
        self.tie_rod_line.set_data([tr_chassis[0], tr_upright[0]], [tr_chassis[1], tr_upright[1]])
        self.tie_rod_line.set_3d_properties([tr_chassis[2], tr_upright[2]])

        # Append wheel‑centre to path and redraw line
        self._path.append(step["wheel_center"])
        path = np.asarray(self._path)
        self.path_line.set_data(path[:, 0], path[:, 1])
        self.path_line.set_3d_properties(path[:, 2])
        plt.draw(); plt.pause(1e-9)

    def display(self):
        plt.ioff(); plt.show()

class Plotter2DBase:
    def __init__(self, char: SCALAR_CHARACTERISTIC):
        self.char = char

        # per-characteristic labels
        titles = {
            SCALAR_CHARACTERISTIC.CAMBER: ("Camber vs Step", "Camber [deg]"),
            SCALAR_CHARACTERISTIC.CASTER: ("Caster vs Step", "Caster [deg]"),
            SCALAR_CHARACTERISTIC.TOE   : ("Toe vs Step",    "Toe [deg]"),
        }
        self._title, self._ylabel = titles[char]

        self._xs: List[int] = []
        self._ys: List[float] = []

        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(4, 3))
        self.ax: plt.Axes = self.fig.add_subplot(111)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel(self._ylabel)
        self.ax.set_title(self._title)
        self.ax.grid(True, linestyle=":", linewidth=0.5)

        self._line, = self.ax.plot([], [], "-o", lw=1.5)

        plt.ion()
        plt.show(block=False)

    def update(self, attitude: Dict[str, float]):
        """Append the current value (deg) taken from *attitude* dict."""
        value = attitude[self.char.name.lower()]      # keys: camber/caster/toe
        self._xs.append(len(self._ys))
        self._ys.append(value)

        self._line.set_data(self._xs, self._ys)
        self.ax.relim(); self.ax.autoscale_view()
        plt.draw()

    def display(self):
        plt.ioff(); plt.show()