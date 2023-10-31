from __future__ import annotations

import argparse
import logging
import random
import sys
import time

from avstack.config import Config
from avstack.utils.decorators import FunctionTriggerIterationMonitor
from PyQt5 import QtCore, QtWidgets

from mate import display, load_scenario_from_config


def do_run(MainThread):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to scenario config file')
    parser.add_argument('--display', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--sleeps', default=0.01)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    cfg_world = Config.fromfile(cfg.world)
    extent = cfg_world.spatial.extent

    if args.display:
        print('Running with display...')
        app = QtWidgets.QApplication(sys.argv)
        window = display.MainWindow(extent=extent, thread=MainThread(cfg, args.sleeps))
        app.exec_()
    else:
        print('Running without display...')
        if args.debug:
            world, objects, agents, commandcenter = \
                load_scenario_from_config(cfg)
            while True:
                _run_inner(
                    thread=None,
                    world=world,
                    objects=objects,
                    agents=agents,
                    commandcenter=commandcenter,
                    sleeps=args.sleeps
                )
        else:
            MainThread(cfg, args.sleeps).run()


@FunctionTriggerIterationMonitor(print_rate=1/2)
def _run_inner(thread, world, objects, agents, commandcenter, sleeps=0.01):
    world.tick()
    random.shuffle(agents)

    # -- agents run sensor fusion pipelines
    for agent in agents:
        agent.tick()

    # -- agents perform planning and movements
    for agent in agents:
        agent.plan()
        agent.move()

    # -- run the central processing
    tracks_out = commandcenter.tick()

    # -- update displays
    if thread is not None:
        thread.truth_signal.emit(world.frame, world.t, world.objects, world.agents)
        track_objs = {track.ID:track for track in tracks_out}
        thread.estim_signal.emit(world.frame, world.t, track_objs, world.agents)
    time.sleep(sleeps)


class MainThread(QtCore.QThread):
    truth_signal = QtCore.pyqtSignal(int, float, object, object)
    detec_signal = QtCore.pyqtSignal(int, float, object, object)
    estim_signal = QtCore.pyqtSignal(int, float, object, object)

    def __init__(self, cfg, sleeps=0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initialize(cfg)
        self.sleeps = sleeps

    def initialize(self, cfg):
        self.world, self.objects, self.agents, self.commandcenter = \
            load_scenario_from_config(cfg)

    def run(self):
        try:
            while True:
                _run_inner(
                    thread=self,
                    world=self.world,
                    objects=self.objects,
                    agents=self.agents,
                    commandcenter=self.commandcenter,
                    sleeps=self.sleeps
                )
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.warning(e, exc_info=True)
        finally:
            pass


if __name__ == "__main__":
    do_run(MainThread)