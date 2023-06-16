import os
#################################################################
# NOTE: These two lines are VERY IMPORTANT -- they ensure qt
# uses its own path to the graphics plugins and not the cv2 path
import cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
#################################################################

import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, extent, thread, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self._thread = thread
        self._thread.signal.connect(self.update_data)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self.extent = extent
        self.objects = {}
        self.agents = {}

        self._plot_ref = None
        self.update_plot()
        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        self._thread.start()

    def update_data(self, objects, agents):
        self.objects = objects
        self.agents = agents

    def update_plot(self):
        # Note: we no longer need to clear the axis.
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.

            # -- plot objects
            for obj_ID, obj in self.objects.items():
                self.canvas.axes.plot(obj.position.x, obj.position.y, 'yo')

            # -- plot agents
            for agent_ID, agent in self.agents.items():
                if agent.is_root:
                    self.canvas.axes.plot(agent.position.x, agent.position.y, 'ro')
                else:
                    self.canvas.axes.plot(agent.position.x, agent.position.y, 'go')

            # plot_refs = self.canvas.axes.plot(self.xdata, self.ydata, 'r')
            # self._plot_ref = plot_refs[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            # self._plot_ref.set_ydata(self.ydata)
            raise NotImplementedError

        # Trigger the canvas to update and redraw.
        self.canvas.draw()