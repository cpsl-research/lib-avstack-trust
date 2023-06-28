import os

#################################################################
# NOTE: These two lines are VERY IMPORTANT -- they ensure qt
# uses its own path to the graphics plugins and not the cv2 path
import cv2


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
#################################################################

import matplotlib


matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5 import QtCore, QtWidgets


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, extent, thread, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self._thread = thread
        self._thread.signal.connect(self.update_data)

        self.canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.initialized = False
        self.setCentralWidget(self.canvas)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self.extent = extent
        self.frame = 0
        self.t = 0
        self.objects = {}
        self.agents = {}
        self.pts = []

        self._plot_ref = None
        self.update_plot()
        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        self._thread.start()

    def update_data(self, frame, t, objects, agents):
        self.frame = frame
        self.t = t
        self.objects = objects
        self.agents = agents

    def update_plot(self, obj_color="y", radicle_color="r", root_color="g"):
        # -- clear axes and set lims
        for pt in self.pts:
            pt.remove()
        self.pts = []
        self.canvas.axes.set_title(f"Frame {self.frame:05d}...Time {self.t:6.2f} s")

        # -- plot objects
        for obj_ID, obj in self.objects.items():
            self.pts.extend(
                self.canvas.axes.plot(obj.position.x[0], obj.position.x[1], obj_color + "o")
            )

        # -- plot agents
        for agent_ID, agent in self.agents.items():
            if agent.is_root:
                self.pts.extend(
                    self.canvas.axes.plot(
                        agent.position.x[0], agent.position.x[1], root_color + "o"
                    )
                )
            else:
                self.pts.extend(
                    self.canvas.axes.plot(
                        agent.position.x[0], agent.position.x[1], radicle_color + "o"
                    )
                )

        # -- legend
        if not self.initialized:
            self.canvas.axes.set_xlim(*self.extent[0])
            self.canvas.axes.set_ylim(*self.extent[1])
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
                markeredgecolor=radicle_color,
                markerfacecolor=radicle_color,
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
            self.canvas.axes.legend(
                handles=[obj_point, rad_point, root_point], loc="upper right"
            )
            self.canvas.axes.set_xlabel('X (index 0)')
            self.canvas.axes.set_ylabel('Y (index 1)')
            self.initialized = True

        # Trigger the canvas to update and redraw.
        self.canvas.draw()
