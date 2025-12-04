# default
from typing import Dict, List
from enum import Enum, auto

# third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink

def _compute_wheel_cylinder(
    wc: np.ndarray,
    step: Dict[str, np.ndarray],
    radius: float,
    width: float,
    n_theta: int = 40,
):
    """
    Compute two circles (front/back) that form a cylinder outline for the wheel.

    wc      : wheel centre (3,)
    step    : kinematic step dict (contains W_* rim points and optionally 'wheel_axis')
    radius  : wheel radius (wr)
    width   : wheel width (ww)
    n_theta : number of points around the circle
    """
    wc = np.asarray(wc, dtype=float)

    # 1) Prefer the true wheel axis if provided by the solver
    axis = None
    if "wheel_axis" in step:
        a = np.asarray(step["wheel_axis"], dtype=float)
        n = np.linalg.norm(a)
        if n > 1e-9:
            axis = a / n

    # 2) Fallback: infer axis from rim points (choose pair with max cross product)
    if axis is None:
        rim_keys = [k for k in step.keys() if k.startswith("W_")]
        rim_vecs = [np.asarray(step[k], float) - wc for k in rim_keys]

        if len(rim_vecs) >= 2:
            best_cross = 0.0
            best_pair = None
            for i in range(len(rim_vecs)):
                for j in range(i + 1, len(rim_vecs)):
                    c = np.cross(rim_vecs[i], rim_vecs[j])
                    n = np.linalg.norm(c)
                    if n > best_cross:
                        best_cross = n
                        best_pair = c
            if best_pair is not None and best_cross > 1e-9:
                axis = best_pair / best_cross

    # 3) If all else fails, just assume global Y as axis
    if axis is None:
        axis = np.array([0.0, 1.0, 0.0])

    # Build an orthonormal basis (u, v, axis)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])

    u = tmp - np.dot(tmp, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    # Centres of front/back rims along the axis
    c_front = wc - 0.5 * width * axis
    c_back  = wc + 0.5 * width * axis

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    circle_front = c_front[:, None] + radius * (u[:, None] * cos_t + v[:, None] * sin_t)
    circle_back  = c_back[:, None]  + radius * (u[:, None] * cos_t + v[:, None] * sin_t)

    return circle_front, circle_back

class SCALAR_CHARACTERISTIC(Enum):
    CAMBER = auto()
    CASTER = auto()
    TOE    = auto()

class AXLE_CHARACTERISTIC(Enum):
    PLUNGE = auto()     # mm
    ANGLE_IB = auto()   # deg
    ANGLE_OB = auto()   # deg

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

        # wheel cylinder outline (two circles: front/back)
        self.wheel_cyl_front = self.ax.plot([], [], [], c="0.5", lw=1.5)[0]
        self.wheel_cyl_back  = self.ax.plot([], [], [], c="0.5", lw=1.5)[0]

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

        # wheel cylinder outline (uses wr, ww, wc and rim points)
        try:
            circle_front, circle_back = _compute_wheel_cylinder(
                wc=wc,
                step=step,
                radius=hp.wr,
                width=hp.ww,
            )

            # front circle
            self.wheel_cyl_front.set_data(circle_front[0, :], circle_front[1, :])
            self.wheel_cyl_front.set_3d_properties(circle_front[2, :])

            # back circle
            self.wheel_cyl_back.set_data(circle_back[0, :], circle_back[1, :])
            self.wheel_cyl_back.set_3d_properties(circle_back[2, :])
        except Exception:
            # If anything goes wrong (no points / numerical issues), clear the cylinder
            self.wheel_cyl_front.set_data([], [])
            self.wheel_cyl_front.set_3d_properties([])
            self.wheel_cyl_back.set_data([], [])
            self.wheel_cyl_back.set_3d_properties([])

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
        self.hp = hp
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
        self.ax.set_ylim(0, 1000)
        self.ax.set_zlim(0, 1000)
        self.ax.set_box_aspect([1, 1, 1])

        # chassis hard-points (fixed in the vehicle frame)
        for pt in (hp.tl_f, hp.ucl_ib, hp.lcl_ib, hp.s_ib):
            self.ax.scatter(*pt, c="black", s=25)

        # dynamic point scatter for the upright / wheel
        self.mov_scatter = self.ax.scatter([], [], [], c="black", s=25)

        # outboard and inboard axle joint – dynamic, shown in red
        self.piv_ob_scatter = self.ax.scatter([], [], [], c="red", s=35, label="axle outboard")
        self.piv_ib_scatter = self.ax.scatter([], [], [], c="red", s=35, label="axle inboard")

        # link lines
        self.utl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.ltl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.ucl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.lcl_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]
        self.shock_line = self.ax.plot([], [], [], c="black", lw=1.5)[0]

        # axle polyline (wc -> pivot outboard -> pivot inboard)
        self.axle_line = self.ax.plot([], [], [], c="red", lw=1.5)[0]

        # wheel-centre path
        self.path_line, = self.ax.plot([], [], [], c="tab:orange", lw=1.5)

        # wheel cylinder outline (front/back circles)
        self.wheel_cyl_front = self.ax.plot([], [], [], c="0.5", lw=1.5)[0]
        self.wheel_cyl_back  = self.ax.plot([], [], [], c="0.5", lw=1.5)[0]

    def update(self, step: Dict[str, np.ndarray]):
        wc = step["wc"]
        piv_ob = step["piv_ob"]
        piv_ib = step["piv_ib"]
        s_ob = step["s_ob"]
        ucl_ob = step["ucl_ob"]
        lcl_ob = step["lcl_ob"]

        # movable scatter points (upright / wheel)
        pts = np.vstack([wc, s_ob, ucl_ob, lcl_ob])
        self.mov_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        # outboard and inboard axle point in red
        self.piv_ob_scatter._offsets3d = ([piv_ob[0]], [piv_ob[1]], [piv_ob[2]])
        self.piv_ib_scatter._offsets3d = ([piv_ib[0]], [piv_ib[1]], [piv_ib[2]])

        # wheel cylinder outline
        try:
            circle_front, circle_back = _compute_wheel_cylinder(
                wc=wc,
                step=step,
                radius=self.hp.wr,
                width=self.hp.ww,
            )

            self.wheel_cyl_front.set_data(circle_front[0, :], circle_front[1, :])
            self.wheel_cyl_front.set_3d_properties(circle_front[2, :])

            self.wheel_cyl_back.set_data(circle_back[0, :], circle_back[1, :])
            self.wheel_cyl_back.set_3d_properties(circle_back[2, :])
        except Exception:
            self.wheel_cyl_front.set_data([], [])
            self.wheel_cyl_front.set_3d_properties([])
            self.wheel_cyl_back.set_data([], [])
            self.wheel_cyl_back.set_3d_properties([])

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

        # axle polyline
        axle_pts = np.vstack([wc, piv_ob, piv_ib])
        self.axle_line.set_data(axle_pts[:, 0], axle_pts[:, 1])
        self.axle_line.set_3d_properties(axle_pts[:, 2])

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
            SCALAR_CHARACTERISTIC.CAMBER: ("Camber vs Shock Travel", "Camber [deg]"),
            SCALAR_CHARACTERISTIC.CASTER: ("Caster vs Shock Travel", "Caster [deg]"),
            SCALAR_CHARACTERISTIC.TOE   : ("Toe vs Shock Travel",    "Toe [deg]"),
        }
        title, ylabel = titles[char]

        # one-liner does all the boiler-plate
        self.fig, self.ax = self._make_axes(
            is_3d=False,
            xlabel="Shock Travel [mm]",
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
            xlabel="Shock Travel [mm]",
            ylabel=f"{axis.name}-coordinate [mm]",
            title=f"{point_key} – {axis.name} vs Shock Travel",
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

class AxleCharacteristicsPlotter(PlotterBase):
    def __init__(self, char: AXLE_CHARACTERISTIC):
        self.char = char
        titles = {
            AXLE_CHARACTERISTIC.PLUNGE:   ("Axle Plunge vs Shock Travel", "Plunge [mm]"),
            AXLE_CHARACTERISTIC.ANGLE_IB: ("Inboard CV Angle vs Shock Travel", "Angle [deg]"),
            AXLE_CHARACTERISTIC.ANGLE_OB: ("Outboard CV Angle vs Shock Travel", "Angle [deg]"),
        }
        title, ylabel = titles[char]

        self.fig, self.ax = self._make_axes(
            is_3d=False,
            xlabel="Shock Travel [mm]",
            ylabel=ylabel,
            title=title,
        )

        self._xs, self._ys = [], []
        self._line, = self.ax.plot([], [], "-o", lw=1.5)

    def update(self, step: Dict[str, np.ndarray]):
        """
        Extract axle data from the step dictionary.
        Requires step['axle_data'] to be present.
        """
        if "axle_data" not in step:
            return 
            
        data = step["axle_data"]
        
        val = 0.0
        if self.char == AXLE_CHARACTERISTIC.PLUNGE:
            val = data["plunge_mm"]
        elif self.char == AXLE_CHARACTERISTIC.ANGLE_IB:
            val = data["angle_ib_deg"]
        elif self.char == AXLE_CHARACTERISTIC.ANGLE_OB:
            val = data["angle_ob_deg"]

        self._xs.append(len(self._ys))
        self._ys.append(val)

        self._line.set_data(self._xs, self._ys)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()