# default
from typing import Dict, List
from enum import Enum, auto

# third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ours
from scripts.hardpoints import DoubleAArm, SemiTrailingLink

class SCALAR_CHARACTERISTIC(Enum):
    CAMBER = auto()
    CASTER = auto()
    TOE    = auto()

class POINT_AXIS(Enum):
    X      = 0
    Y      = 1
    Z      = 2

class PlotterBase:
    @staticmethod
    def _make_axes(
        *,
        is_3d: bool,
        figsize=(4, 3),
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str | None = None,
        title: str | None = None,
    ):
        fig = plt.figure(figsize=figsize)
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel or "")
        else:
            ax = fig.add_subplot(111)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.5)

        plt.ion()
        plt.show(block=False)
        return fig, ax

    def display(self):
        plt.ioff()
        plt.show()

    def update(self, *args, **kwargs):
        raise NotImplementedError
    
class DoubleAArmPlotter(PlotterBase):
    def __init__(self, hp: DoubleAArm):
        self.hp = hp
        self._path = []

        # common 3-D axis creation
        self.fig, self.ax = self._make_axes(
            is_3d=True,
            figsize=(7, 7),
            xlabel="X [mm]",
            ylabel="Y [mm]",
            zlabel="Z [mm]",
            title="Double-A-Arm Corner",
        )
        self.ax.view_init(elev=20, azim=30)

        self._init_chassis_artists()
    
    def _init_chassis_artists(self):
        hp = self.hp

        # uniform cube axes
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 1000)
        self.ax.set_zlim(0, 1000)
        self.ax.set_box_aspect([1, 1, 1])

        # static inboard a arm points
        self.ax.scatter(hp.uf[0], hp.uf[1], hp.uf[2], color='black', s=25)
        self.ax.scatter(hp.ur[0], hp.ur[1], hp.ur[2], color='black', s=25)
        self.ax.scatter(hp.lf[0], hp.lf[1], hp.lf[2], color='black', s=25)
        self.ax.scatter(hp.lr[0], hp.lr[1], hp.lr[2], color='black', s=25)

        # static inboard a arm lines
        for pt in (hp.uf, hp.ur, hp.lf, hp.lr):
            self.ax.scatter(pt[0], pt[1], pt[2], c="black", s=25)

        self.ax.scatter(hp.s_ib[0], hp.s_ib[1], hp.s_ib[2], c="black", s=25)

        # dynamic ball joint points
        self.mov_scatter = self.ax.scatter([], [], [], c="black", s=25)

        # dynamic a arm lines
        self.upper_lines = [self.ax.plot([], [], [], c="black", lw=1.5)[0] for _ in range(2)]
        self.lower_lines = [self.ax.plot([], [], [], c="black", lw=1.5)[0] for _ in range(2)]
        self.shock_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.tie_rod_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]

        # path line for wheel center
        self.path_line, = self.ax.plot([], [], [], c="red", lw=1.5)

        # dynamic points for the wheel rim
        self.wheel_scatter = self.ax.scatter([], [], [], c="tab:blue", s=15, label="wheel-rim pts")

        plt.ion()
        plt.show(block=False)

    def update(self, step: Dict[str, np.ndarray]):
        hp = self.hp
        lbj = step["lbj"]
        ubj = step["ubj"]
        wc  = step["wc"]
        sha = step["s_ob"]
        tr_chassis = step["tr_ib"]
        tr_upright = step["tr_ob"]

        pts = np.vstack([lbj, ubj, wc, sha])
        self.mov_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        # wheel rim points
        wheel_pts = [v for k, v in step.items() if k.startswith("W_")]
        if wheel_pts:                       # solver found rim points
            wp = np.vstack(wheel_pts)
            self.wheel_scatter._offsets3d = (wp[:, 0], wp[:, 1], wp[:, 2])
        else:                               # outside travel range, etc.
            self.wheel_scatter._offsets3d = ([], [], [])

        # upper front‑rear
        for line, chassis in zip(self.upper_lines, (hp.uf, hp.ur)):
            line.set_data([chassis[0], ubj[0]], [chassis[1], ubj[1]])
            line.set_3d_properties([chassis[2], ubj[2]])

        # lower front‑rear
        for line, chassis in zip(self.lower_lines, (hp.lf, hp.lr)):
            line.set_data([chassis[0], lbj[0]], [chassis[1], lbj[1]])
            line.set_3d_properties([chassis[2], lbj[2]])

        # shock
        self.shock_line.set_data([hp.s_ib[0], sha[0]], [hp.s_ib[1], sha[1]])
        self.shock_line.set_3d_properties([hp.s_ib[2], sha[2]])

        # tie rod
        self.tie_rod_line.set_data([tr_chassis[0], tr_upright[0]], [tr_chassis[1], tr_upright[1]])
        self.tie_rod_line.set_3d_properties([tr_chassis[2], tr_upright[2]])

        # Append wheel‑centre to path and redraw line
        self._path.append(step["wc"])
        path = np.asarray(self._path)
        self.path_line.set_data(path[:, 0], path[:, 1])
        self.path_line.set_3d_properties(path[:, 2])

        plt.draw()
        plt.pause(1e-9)

class SemiTrailingLinkPlotter(PlotterBase):
    def __init__(self, hp: SemiTrailingLink):
        self.hp   = hp
        self._path: list[np.ndarray] = []

        self.fig, self.ax = self._make_axes(
            is_3d=True,
            figsize=(7, 7),
            xlabel="X [mm]",
            ylabel="Y [mm]",
            zlabel="Z [mm]",
            title="Semi-Trailing-Link Corner",
        )
        self.ax.view_init(elev=20, azim=30)

        self._init_static_artists()

    def _init_static_artists(self):
        hp = self.hp

        # uniform cube axes
        self.ax.set_xlim(1000, 2000)
        self.ax.set_ylim(   0, 1000)
        self.ax.set_zlim(   0, 1000)
        self.ax.set_box_aspect([1, 1, 1])

        # chassis hard-points
        for pt in (hp.tl_f, hp.ucl_ib, hp.lcl_ib, hp.s_ib):
            self.ax.scatter(*pt, c="black", s=25)

        # dynamic point scatter
        self.mov_scatter = self.ax.scatter([], [], [], c="black", s=25)

        # link lines
        self.utl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.ltl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.ucl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.lcl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.shock_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]

        # wheel-centre path
        self.path_line, = self.ax.plot([], [], [], c="red", lw=1.5)

        # dynamic points for the wheel rim
        self.wheel_scatter = self.ax.scatter([], [], [], c="tab:blue", s=15, label="wheel-rim pts")

    def update(self, step: Dict[str, np.ndarray]):
        wc = step["wc"]
        piv_ob = step["piv_ob"]
        s_ob = step["s_ob"]
        ucl_ob = step["ucl_ob"]
        lcl_ob = step["lcl_ob"]

        # movable scatter points
        pts = np.vstack([wc, s_ob, ucl_ob, lcl_ob])
        self.mov_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        # wheel rim points
        wheel_pts = [v for k, v in step.items() if k.startswith("W_")]
        if wheel_pts: # solver found rim points
            wp = np.vstack(wheel_pts)
            self.wheel_scatter._offsets3d = (wp[:, 0], wp[:, 1], wp[:, 2])
        else: # outside travel range, etc.
            self.wheel_scatter._offsets3d = ([], [], [])

        # upper trailing link
        self.utl_line.set_data([step["tl_f"][0], ucl_ob[0]],
                              [step["tl_f"][1], ucl_ob[1]])
        self.utl_line.set_3d_properties([step["tl_f"][2], ucl_ob[2]])

        # lower trailing link
        self.ltl_line.set_data([step["tl_f"][0], lcl_ob[0]],
                              [step["tl_f"][1], lcl_ob[1]])
        self.ltl_line.set_3d_properties([step["tl_f"][2], lcl_ob[2]])

        # upper camber link
        self.ucl_line.set_data([step["ucl_ib"][0], ucl_ob[0]],
                               [step["ucl_ib"][1], ucl_ob[1]])
        self.ucl_line.set_3d_properties([step["ucl_ib"][2], ucl_ob[2]])

        # lower camber link
        self.lcl_line.set_data([step["lcl_ib"][0], lcl_ob[0]],
                               [step["lcl_ib"][1], lcl_ob[1]])
        self.lcl_line.set_3d_properties([step["lcl_ib"][2], lcl_ob[2]])

        # shock
        self.shock_line.set_data([step["s_ib"][0], s_ob[0]],
                                 [step["s_ib"][1], s_ob[1]])
        self.shock_line.set_3d_properties([step["s_ib"][2], s_ob[2]])

        # wheel-centre path
        self._path.append(wc.copy())
        path = np.asarray(self._path)
        self.path_line.set_data(path[:, 0], path[:, 1])
        self.path_line.set_3d_properties(path[:, 2])

        plt.draw()
        plt.pause(1e-9)

class CharacteristicPlotter(PlotterBase):
    def __init__(self, char: SCALAR_CHARACTERISTIC):
        self.char = char
        titles = {
            SCALAR_CHARACTERISTIC.CAMBER: ("Camber vs Step", "Camber [deg]"),
            SCALAR_CHARACTERISTIC.CASTER: ("Caster vs Step", "Caster [deg]"),
            SCALAR_CHARACTERISTIC.TOE   : ("Toe vs Step",    "Toe [deg]"),
        }
        title, ylabel = titles[char]

        # one-liner does all the boiler-plate
        self.fig, self.ax = self._make_axes(
            is_3d=False,
            xlabel="Step",
            ylabel=ylabel,
            title=title,
        )

        self._xs, self._ys = [], []
        self._line, = self.ax.plot([], [], "-o", lw=1.5)

    def update(self, attitude: Dict[str, float]):
        """Append the current value (deg) taken from *attitude* dict."""
        value = attitude[self.char.name.lower()]
        self._xs.append(len(self._ys))
        self._ys.append(value)

        self._line.set_data(self._xs, self._ys)
        self.ax.relim(); self.ax.autoscale_view()

        plt.draw()

class PointPlotter(PlotterBase):
    def __init__(self, point_key: str, axis: POINT_AXIS):
        self.point_key = point_key
        self.coord_idx = axis.value # 0 / 1 / 2

        self.fig, self.ax = self._make_axes(
            is_3d=False,
            xlabel="Step",
            ylabel=f"{axis.name}-coordinate [mm]",
            title=f"{point_key} – {axis.name} vs Step",
        )

        self._xs, self._ys = [], []
        self._line, = self.ax.plot([], [], "-o", lw=1.5)

    def update(self, step: Dict[str, np.ndarray]):
        if self.point_key not in step:
            raise KeyError(f"PointPlotter: '{self.point_key}' not present in step.")

        value = float(step[self.point_key][self.coord_idx])

        self._xs.append(len(self._ys))
        self._ys.append(value)

        self._line.set_data(self._xs, self._ys)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()