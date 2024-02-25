import itertools
from typing import Tuple

import numpy as np
from avstack.config import GEOMETRY, MODELS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import Position, ReferenceFrame
from avstack.modules.perception.detections import CentroidDetection


class SensorModel:
    _ids = itertools.count()

    def __init__(
        self,
        x,
        q,
        noise,
        reference,
        extent,
        fov,
        ID=None,
        Pd=0.95,
        Dfa=1e-6,
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
        self.extent = extent
        self.fov = GEOMETRY.build(fov)
        self._fov_area_uniform = self.fov.radius**2
        self.noise = noise
        self._reference = ReferenceFrame(x=x, q=q, reference=reference)

    def as_reference(self):
        return self._reference

    def __call__(self, frame, timestamp, objects):
        # -- add false positives via rejection sampling
        n_fp = np.random.poisson(self.Dfa * self._fov_area_uniform)
        x = np.random.uniform(low=-self.fov.radius, high=self.fov.radius, size=n_fp)
        y = np.random.uniform(low=-self.fov.radius, high=self.fov.radius, size=n_fp)
        z = np.zeros((n_fp,))
        vs = np.column_stack((x, y, z))
        if len(vs) > 0:
            fov_test_fps = self.fov.check_point(vs.T)
            objs_fp = [Position(v, self._reference) for v in vs[fov_test_fps, :]]
        else:
            objs_fp = []

        # -- check objects in fov
        obj_xs = np.array([obj.position.x for obj in objects])
        fov_test_objs = self.fov.check_point(obj_xs.T)
        obj_in_view = [obj for obj, in_fov in zip(objects, fov_test_objs) if in_fov]

        # -- make measurements
        detections = []
        for i, objs in enumerate((obj_in_view, objs_fp)):
            for obj in objs:
                if not isinstance(obj, Position):
                    obj = obj.position
                if (i == 0) and (np.random.rand() > self.Pd):
                    continue  # sometimes false negative
                detections.append(self.observe(self, obj))
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
        Pd=0.95,
        Dfa=1e-6,
        name: str = "sensor",
    ) -> None:
        super().__init__(
            x, q, noise, reference, extent, fov, ID=ID, Pd=Pd, Dfa=Dfa, name=name
        )

    def observe(self, sensor, obj, noisy=True):
        """Make local observation and add Gaussian noise"""
        if not isinstance(obj, Position):
            obj = obj.position
        obj = obj.change_reference(self.as_reference(), inplace=False)
        xyz = obj.x + self.noise * np.random.randn(3) if noisy else obj.x
        detection = CentroidDetection(
            source_identifier=sensor.ID,
            centroid=xyz,
            reference=self.as_reference(),
        )
        return detection
