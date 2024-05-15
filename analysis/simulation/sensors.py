import itertools
from typing import TYPE_CHECKING, Tuple


if TYPE_CHECKING:
    from avstack.geometry import Shape


import numpy as np
from avstack.config import GEOMETRY, MODELS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import Position, ReferenceFrame
from avstack.modules.perception.detections import CentroidDetection


class SensorModel:
    _ids = itertools.count()

    def __init__(
        self,
        x: np.ndarray,
        q: np.quaternion,
        noise: np.ndarray,
        reference: ReferenceFrame,
        extent: Tuple,
        fov: "Shape",
        ID=None,
        Pd: float = 0.95,
        Dfa: float = 1e-6,
        Pp_FP: float = 0,
        Pp_FN: float = 0,
        name="sensor",
    ) -> None:
        """Base class for an observation model

        noise  - component-wise noise model
        extent - the area of the ENTIRE space (todo...update based on fov later maybe)
        fov    - the angular field of view of the sensor (max range, half azimuth, half elevation)
        Pd     - probability of detection of a true object
        Dfa    - density of false alarms in 1/m^2
        """
        self.name = name
        self.ID = ID if ID else next(SensorModel._ids)
        self.x = x
        self.q = q
        self.Pd = Pd
        self.Dfa = Dfa
        self.Pp_FP = Pp_FP
        self.Pp_FN = Pp_FN
        self._last_fp = []
        self._last_fn = {}
        self.extent = extent
        self.fov = GEOMETRY.build(fov)
        self._fov_area_uniform = self.fov.radius**2
        self.noise = noise
        self._reference = ReferenceFrame(x=x, q=q, reference=reference)

    def as_reference(self):
        return self._reference

    def __call__(self, frame, timestamp, objects):
        # filter objs in FOV
        obj_xs = np.array(
            [
                (obj.position.change_reference(self._reference, inplace=False)).x
                for obj in objects
            ]
        )
        fov_test_objs = self.fov.check_point(obj_xs.T)
        objs_in_view = [obj for obj, in_fov in zip(objects, fov_test_objs) if in_fov]
        IDs = {obj.ID for obj in objs_in_view}

        # remove false negatives with persistence
        self._last_fn = {ID for ID in self._last_fn if ID in IDs}
        n_fn_persist = np.random.binomial(n=len(self._last_fn), p=self.Pp_FN)
        n_fn_new = max(
            0, np.random.binomial(len(objs_in_view) - n_fn_persist, p=1 - self.Pd)
        )
        if len(self._last_fn) > 0:
            fns_persist = np.random.choice(
                self._last_fn,
                size=n_fn_persist,
                replace=False,
            )
        else:
            fns_persist = []
        fns_new = list(
            np.random.choice(
                list(IDs.difference(fns_persist)), size=n_fn_new, replace=False
            )
        )
        fns = fns_persist + fns_new
        for idx in sorted(np.argwhere(IDs == fns), reverse=True):
            del objs_in_view[idx]

        # add false positives with persistence
        n_fp_persist = np.random.binomial(n=len(self._last_fp), p=self.Pp_FP)
        n_fp_new = max(0, np.random.poisson(self.Dfa * self.fov.area) - n_fp_persist)
        # TODO: "could not convert object to sequence" error
        fps_persist = list(
            np.random.choice(self._last_fp, size=n_fp_persist, replace=False)
        )
        fps_new = [
            Position(self.fov.sample_point(), self._reference) for _ in range(n_fp_new)
        ]
        fps = fps_persist + fps_new
        objs_in_view += fps
        self._last_fp = fps

        # make measurements
        detections = []
        for obj in objs_in_view:
            detections.append(self.observe(obj))
        return DataContainer(frame, timestamp, detections, self.ID)

    def observe(self, sensor, obj, noisy):
        raise NotImplementedError


@MODELS.register_module()
class PositionSensor(SensorModel):
    def __init__(
        self,
        extent: Tuple[Tuple, Tuple, Tuple],
        reference: ReferenceFrame,
        x: np.ndarray = np.zeros((3,)),
        q: np.quaternion = np.quaternion(1),
        noise: np.ndarray = np.zeros((3,)),
        fov: ConfigDict = {"type": "Circle", "radius": 20},
        ID=None,
        Pd: float = 0.95,
        Dfa: float = 1e-6,
        Pp_FP: float = 0,
        Pp_FN: float = 0,
        name: str = "sensor",
    ) -> None:
        super().__init__(
            x,
            q,
            noise,
            reference,
            extent,
            fov,
            ID=ID,
            Pd=Pd,
            Dfa=Dfa,
            Pp_FP=Pp_FP,
            Pp_FN=Pp_FN,
            name=name,
        )

    def observe(self, obj, noisy=True):
        """Make local observation and add Gaussian noise"""
        if not isinstance(obj, Position):
            obj = obj.position
        obj = obj.change_reference(self.as_reference(), inplace=False)
        xyz = obj.x + self.noise * np.random.randn(3) if noisy else obj.x
        detection = CentroidDetection(
            source_identifier=self.ID,
            centroid=xyz,
            reference=self.as_reference(),
        )
        return detection
