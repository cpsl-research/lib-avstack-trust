from __future__ import annotations

import argparse
import cProfile
import logging
import random
import sys
import time

import avstack  # noqa # pylint: disable=unused-import
from avstack.config import AGENTS, MODELS, Config
from avstack.utils.decorators import FunctionTriggerIterationMonitor
from PyQt5 import QtCore, QtWidgets
from simulation import display

import mate  # noqa # pylint: disable=unused-import


def load_scenario_from_config_file(filename: str):
    cfg = Config.fromfile(filename=filename)
    return load_scenario_from_config(cfg)


def load_scenario_from_config(cfg, world=None):
    # -- world
    if world is None:
        world = MODELS.build(cfg.world)

    # -- objects
    objects = []
    for cfg_obj in cfg.objects:
        obj = AGENTS.build(cfg_obj, default_args={"world": world})
        objects.append(obj)
        world.add_object(obj)

    # -- agents
    agents = []
    for cfg_agent in cfg.agents:
        agent = AGENTS.build(cfg_agent, default_args={"world": world})
        agents.append(agent)
        world.add_agent(agent)

    # -- command center
    commandcenter = AGENTS.build(cfg.commandcenter, default_args={"world": world})

    return world, objects, agents, commandcenter


def do_run(MainThread):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to scenario config file")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sleeps", default=0.0001)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    world = MODELS.build(cfg.world)
    extent = world.extent

    if args.display:
        print("Running with display...")
        app = QtWidgets.QApplication(sys.argv)
        window = display.MainWindow(extent=extent, thread=MainThread(cfg, args.sleeps))
        app.exec_()
    else:
        print("Running without display...")
        if args.debug:
            world, objects, agents, commandcenter = load_scenario_from_config(cfg)
            while True:
                _run_inner(
                    thread=None,
                    world=world,
                    objects=objects,
                    agents=agents,
                    commandcenter=commandcenter,
                    sleeps=args.sleeps,
                )
        else:
            MainThread(cfg, args.sleeps, world=world).run()


@FunctionTriggerIterationMonitor(print_rate=1 / 2)
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
    output = commandcenter.tick()

    # -- update displays
    if thread is not None:
        thread.truth_signal.emit(
            world.frame, world.timestamp, world.objects, world.agents
        )
        # thread.estim_signal.emit(world.frame, world.timestamp, tracks_out, world.agents)
        # thread.trust_signal.emit(world.frame, world.timestamp, cluster_trusts, agent_trusts)
    time.sleep(sleeps)


class MainThread(QtCore.QThread):
    truth_signal = QtCore.pyqtSignal(int, float, object, object)
    detec_signal = QtCore.pyqtSignal(int, float, object, object)
    estim_signal = QtCore.pyqtSignal(int, float, object, object)
    trust_signal = QtCore.pyqtSignal(int, float, object, object)

    def __init__(self, cfg, sleeps=0.01, world=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sleeps = sleeps
        self.world = world
        self.initialize(cfg)

    def initialize(self, cfg):
        (
            self.world,
            self.objects,
            self.agents,
            self.commandcenter,
        ) = load_scenario_from_config(cfg, world=self.world)

    def run(self):
        try:
            while True:
                _run_inner(
                    thread=self,
                    world=self.world,
                    objects=self.objects,
                    agents=self.agents,
                    commandcenter=self.commandcenter,
                    sleeps=self.sleeps,
                )
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.warning(e, exc_info=True)
        finally:
            pass


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    do_run(MainThread)
    pr.disable()
    pr.dump_stats("last_run.prof")
