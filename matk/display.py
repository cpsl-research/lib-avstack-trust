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
        self.axes = fig.subplots(1,2)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, extent, thread, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self._thread = thread
        self._thread.truth_signal.connect(self.update_truth_data)
        self._thread.estim_signal.connect(self.update_estim_data)

        self.canvas = MplCanvas(self, width=14, height=8, dpi=100)
        self.setCentralWidget(self.canvas)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self.extent = extent
        self.truth = {'supertitle':'Truth', 'initialized':False, 'frame':0, 't':0, 'objects':{}, 'agents':{}, 'pts':[]}
        self.estim = {'supertitle':'Estimated', 'initialized':False, 'frame':0, 't':0, 'objects':{}, 'agents':{}, 'pts':[]}

        self._plot_ref = None
        self.update_plots()
        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()
        self._thread.start()

    @staticmethod
    def _update_data(datastruct, frame, t, objects, agents):
        datastruct['frame'] = frame
        datastruct['t'] = t
        datastruct['objects'] = objects
        datastruct['agents'] = agents
        return datastruct

    @staticmethod
    def _update_plot(extent, axis, datastruct, obj_color, root_color, rad_color):

        # -- clear axes and set lims
        for pt in datastruct['pts']:
            pt.remove()
        datastruct['pts'] = []
        axis.set_title(f"{datastruct['supertitle']}\nFrame {datastruct['frame']:05d}...Time {datastruct['t']:6.2f} s")

        # -- plot objects
        for obj_ID, obj in datastruct['objects'].items():
            datastruct['pts'].extend(
                axis.plot(obj.position.x[0], obj.position.x[1], obj_color + "o")
            )

        # -- plot agents
        for agent_ID, agent in datastruct['agents'].items():
            if agent.is_root:
                datastruct['pts'].extend(
                    axis.plot(
                        agent.position.x[0], agent.position.x[1], root_color + "o"
                    )
                )
            else:
                datastruct['pts'].extend(
                    axis.plot(
                        agent.position.x[0], agent.position.x[1], rad_color + "o"
                    )
                )

        # -- legend
        if not datastruct['initialized']:
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
            axis.legend(
                handles=[obj_point, rad_point, root_point], loc="upper right"
            )
            axis.set_xlabel('X (index 0)')
            axis.set_ylabel('Y (index 1)')
            datastruct['initialized'] = True

    def update_truth_data(self, frame, t, objects, agents):
        self.truth = self._update_data(self.truth, frame, t, objects, agents)

    def update_estim_data(self, frame, t, objects, agents):
        self.estim = self._update_data(self.estim, frame, t, objects, agents)

    def update_plots(self, obj_color="y", rad_color="r", root_color="g"):
        self._update_plot(self.extent, self.canvas.axes[0], self.truth, obj_color, root_color, rad_color)
        self._update_plot(self.extent, self.canvas.axes[1], self.estim, obj_color, root_color, rad_color)
        self.canvas.draw()
