import os


#################################################################
# NOTE: These two lines are VERY IMPORTANT -- they ensure qt
# uses its own path to the graphics plugins and not the cv2 path


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
#################################################################

import matplotlib
import numpy as np


matplotlib.use("Qt5Agg")
from avstack.datastructs import PriorityQueue
from avstack.geometry import GlobalOrigin3D, transform_orientation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge as mWedge
from PyQt5 import QtCore, QtWidgets


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=16, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.subplots(1, 4, width_ratios=[4, 4, 4, 1])
        for ax in self.axes[:3]:
            ax.set_aspect("equal")
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, extent, thread, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self._thread = thread
        self._thread.truth_signal.connect(self.update_truth_data)
        self._thread.agent_signal.connect(self.update_agent_data)
        self._thread.detec_signal.connect(self.update_detect_data)
        self._thread.command_signal.connect(self.update_command_data)
        self._thread.trust_signal.connect(self.update_trust_data)

        self.canvas = MplCanvas(self, width=20, height=8, dpi=100)
        self.setCentralWidget(self.canvas)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self.extent = extent
        self.truth = {
            "supertitle": "Truth",
            "show_fov": True,
            "show_detects": False,
            "initialized": False,
            "frame": 0,
            "t": 0,
            "objects": {},
            "agents": {},
            "pts": [],
            "wedges": [],
        }
        self.agents = {
            "supertitle": "Agents",
            "show_fov": True,
            "show_detects": True,
            "initialized": False,
            "frame": 0,
            "t": 0,
            "objects": {},
            "agents": {},
            "pts": [],
            "wedges": [],
        }
        self.commandcenter = {
            "supertitle": "Command Center",
            "show_fov": True,
            "show_detects": False,
            "initialized": False,
            "frame": 0,
            "t": 0,
            "objects": {},
            "agents": {},
            "pts": [],
            "wedges": [],
        }
        self.detect = {}
        self.detect_frame_skip = 0
        self.detect_frame_buffer = 20
        self.trusts = {
            "supertitle": "Trusts",
            "frame": 0,
            "t": 0,
            "objects": {},
            "agents": {},
            "lines": [],
            "initialized": False,
        }

        self._plot_ref = None
        self.update_plots()
        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()
        self._thread.start()

    def _update_detections(self, frame, t, obj, detections):
        if obj.ID not in self.detect:
            self.detect[obj.ID] = PriorityQueue(
                max_size=None, max_heap=False, empty_returns_none=True
            )
        if (len(self.detect[obj.ID]) == 0) or (
            frame - self.detect[obj.ID].top()[0] > self.detect_frame_skip
        ):
            detects = []
            print(len(detections))
            for det in detections:
                det = det.change_reference(GlobalOrigin3D, inplace=False)
                detects.append((det.xy[0], det.xy[1]))
            self.detect[obj.ID].push(frame, detects)

    @staticmethod
    def _update_data(datastruct, frame, t, objects, agents):
        datastruct["frame"] = frame
        datastruct["t"] = t
        datastruct["objects"] = objects
        datastruct["agents"] = agents

    @staticmethod
    def _update_scenario_plot(
        extent,
        axis,
        datastruct,
        x_data,
        obj_color,
        root_color,
        rad_color,
        x_frame_buffer,
    ):

        # -- clear axes and set lims
        for pt in datastruct["pts"]:
            pt.remove()
        datastruct["pts"] = []
        for wedge in datastruct["wedges"]:
            wedge.remove()
        datastruct["wedges"] = []
        axis.set_title(
            f"{datastruct['supertitle']}\nFrame {datastruct['frame']:05d}...Time {datastruct['t']:6.2f} s"
        )

        # -- plot objects
        for group, objs in datastruct["objects"].items():
            for obj in objs:
                obj.change_reference(GlobalOrigin3D, inplace=True)
                datastruct["pts"].extend(
                    axis.plot(obj.position.x[0], obj.position.x[1], obj_color + ".")
                )

        # -- plot agents
        for agent in datastruct["agents"]:
            color = root_color if agent.trusted else rad_color
            datastruct["pts"].extend(
                axis.plot(agent.position.x[0], agent.position.x[1], color + "*")
            )
            # -- fov wedge
            if datastruct["show_fov"]:
                sens = agent.sensing[
                    list(agent.sensing.keys())[0]
                ]  # HACK for only 1 sensor
                s_ref = sens.as_reference()
                s_global = s_ref.integrate(start_at=GlobalOrigin3D)
                center = [s_global.x[0], s_global.x[1]]
                angle_init = transform_orientation(s_global.q, "quat", "euler")[2]
                wedge = mWedge(
                    center,
                    sens.fov.radius,
                    (angle_init + sens.fov.angle_start) * 180 / np.pi,
                    (angle_init + sens.fov.angle_stop) * 180 / np.pi,
                    alpha=0.3,
                    color=color,
                )
                axis.add_patch(wedge)
                datastruct["wedges"].append(wedge)

            # -- detections
            if datastruct["show_detects"]:
                if agent.ID in x_data:
                    _ = x_data[agent.ID].pop_all_below(
                        datastruct["frame"] - x_frame_buffer
                    )
                    for frame, objs in x_data[agent.ID].heap:
                        f_frac = min(
                            1.0, 1 - (datastruct["frame"] - frame) / x_frame_buffer
                        )
                        for obj in objs:
                            datastruct["pts"].extend(
                                axis.plot(obj[0], obj[1], "kx", alpha=f_frac)
                            )

        # -- legend
        if not datastruct["initialized"]:
            axis.set_xlim(*extent[0])
            axis.set_ylim(*extent[1])
            obj_point = Line2D(
                [0],
                [0],
                label="Object",
                marker="s",
                markersize=10,
                markeredgecolor=obj_color,
                markerfacecolor=obj_color,
                linestyle="",
            )
            rad_point = Line2D(
                [0],
                [0],
                label="Radicle",
                marker="s",
                markersize=10,
                markeredgecolor=rad_color,
                markerfacecolor=rad_color,
                linestyle="",
            )
            root_point = Line2D(
                [0],
                [0],
                label="Root",
                marker="s",
                markersize=10,
                markeredgecolor=root_color,
                markerfacecolor=root_color,
                linestyle="",
            )
            axis.legend(handles=[obj_point, rad_point, root_point], loc="upper right")
            axis.set_xlabel("X (index 0)")
            axis.set_ylabel("Y (index 1)")
            datastruct["initialized"] = True

    @staticmethod
    def _update_trust_plot(
        agent_axis,
        cluster_axis,
        datastruct,
        n_points_eval=200,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=3.0,
    ):
        for ax in [agent_axis, cluster_axis]:
            # -- clear axes and set lims
            ax.set_title(
                f"{datastruct['supertitle']}\nFrame {datastruct['frame']:05d}...Time {datastruct['t']:6.2f} s"
            )
            for line in ax.get_lines():
                line.remove()

        # -- agent trusts
        x = np.linspace(xmin, xmax, n_points_eval)
        for ID, agent_trust in datastruct["agents"].items():
            y = agent_trust.dist.pdf(x)
            line = agent_axis.plot(x, y, label="Agent %i".format(ID))

        # -- cluster trusts
        for ID, cluster_trust in datastruct["objects"].items():
            y = cluster_trust.dist.pdf(x)
            line = cluster_axis.plot(x, y, label="Cluster %i".format(ID))

        # -- legend
        if not datastruct["initialized"]:
            for ax in [agent_axis, cluster_axis]:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

    def update_truth_data(self, frame, t, objects, agents):
        self._update_data(self.truth, frame, t, objects, agents)

    def update_agent_data(self, frame, t, objects, agents):
        self._update_data(self.agents, frame, t, objects, agents)

    def update_detect_data(self, frame, t, obj, detections):
        self._update_detections(frame, t, obj, detections)

    def update_command_data(self, frame, t, objects, agents):
        self._update_data(self.commandcenter, frame, t, objects, agents)

    def update_trust_data(self, frame, t, cluster_trusts, agent_trusts):
        self._update_data(self.trusts, frame, t, cluster_trusts, agent_trusts)

    def update_plots(self, obj_color="y", rad_color="r", root_color="g"):
        # -- plot of the truth
        self._update_scenario_plot(
            self.extent,
            self.canvas.axes[0],
            self.truth,
            {},
            obj_color,
            root_color,
            rad_color,
            x_frame_buffer=0,
        )
        # -- plot of the estimated/detected from agents
        self._update_scenario_plot(
            self.extent,
            self.canvas.axes[1],
            self.agents,
            self.detect,
            obj_color,
            root_color,
            rad_color,
            x_frame_buffer=self.detect_frame_buffer,
        )
        # -- plot of the estimated/detected from cc
        self._update_scenario_plot(
            self.extent,
            self.canvas.axes[2],
            self.commandcenter,
            {},
            obj_color,
            root_color,
            rad_color,
            x_frame_buffer=0,
        )

        # -- plot of the trusts
        # self._update_trust_plot(
        #     self.canvas.axes[2],
        #     self.canvas.axes[3],
        #     self.trusts,
        # )

        self.canvas.draw()
