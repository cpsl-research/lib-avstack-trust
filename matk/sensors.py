import itertools
import numpy as np
from avstack.geometry.transformations import cartesian_to_spherical
from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D, Position, ReferenceFrame
from avstack.modules.perception.detections import RazDetection


class SensorModel():
    _ids = itertools.count()

    def __init__(self, x, q, reference, noise, extent, fov, Pd=0.95, Dfa=1e-6) -> None:
        """Base class for an observation model
        
        noise  - component-wise noise model
        extent - the area of the ENTIRE space (todo...update based on fov later maybe)
        fov    - the angular field of view of the sensor (max range, half azimuth, half elevation)
        Pd     - probability of detection of a true object
        Dfa    - density of false alarms in 1/m^2
        """
        self.ID = next(SensorModel._ids)
        self.x = x
        self.q = q
        self.Pd = Pd
        self.Dfa = Dfa
        self.extent = extent
        self.fov = fov
        self.noise = noise
        self.area = (extent[0][1] - extent[0][0]) * (extent[1][1] - extent[1][0])
        self._reference = ReferenceFrame(x=x, q=q, reference=reference)

    def as_reference(self):
        return self._reference

    def __call__(self, frame, timestamp, objects):
        # -- add false positives
        n_fp = np.random.poisson(self.Dfa * self.area)
        x = np.random.uniform(low=self.extent[0][0], high=self.extent[0][1], size=n_fp)
        y = np.random.uniform(low=self.extent[0][0], high=self.extent[0][1], size=n_fp)
        z = np.zeros((n_fp,))
        objs_fp = [Position(v, GlobalOrigin3D) for v in np.column_stack((x,y,z))]

        # -- make measurements
        detections = []
        for i, objs in enumerate((list(objects.values()), objs_fp)):
            for obj in objs:
                # -- check in fov
                if not isinstance(obj, Position):
                    obj = obj.position
                obj = obj.change_reference(self.as_reference(), inplace=False)
                razel = cartesian_to_spherical(obj.x)
                in_range = razel[0] <= self.fov[0]
                razel[1] = razel[1] % (2*np.pi)
                in_azimuth = min(razel[1], 2*np.pi - razel[1]) <= self.fov[1]
                if not (in_range and in_azimuth):
                    continue
                # -- make measurement
                if (i == 0) and (np.random.rand() > self.Pd):
                    continue  # sometimes false negative
                detections.append(self.observe(self, obj))
        return DataContainer(frame, timestamp, detections, self.ID)
    
    def observe(self, sensor, obj, noisy):
        raise NotImplementedError
    

class PositionSensor(SensorModel):
    def __init__(self, x, q, reference, noise, extent, fov, Pd=0.95, Dfa=1e-6) -> None:
        super().__init__(x, q, reference, noise, extent, fov, Pd=Pd, Dfa=Dfa)

    def observe(self, sensor, obj, noisy=True):
        """Make local observation and add Gaussian noise"""
        if not isinstance(obj, Position):
            obj = obj.position
        obj = obj.change_reference(self.as_reference(), inplace=False)
        xyz = obj.x + self.noise * np.random.randn(3) if noisy else obj.x
        razel = cartesian_to_spherical(xyz)
        detection = RazDetection(
            source_identifier=sensor.ID,
            raz=razel[:2],
            reference=self.as_reference(),
        )
        return detection