import itertools
from typing import Dict, FrozenSet, Union

import numpy as np
from avstack.sensors import LidarData


def get_disjoint_fov_subsets(
    fovs: Dict[int, "FieldOfView"]
) -> Dict[FrozenSet[int], "FieldOfView"]:
    """compute all disjoint FOV subsets from all agents

    A disjoint subsets are subsets that have no overlap with each other.

    Computing all subsets amounts to looping over the 2^n number of
    distinct agent combinations and checking for FOV overlap. Some
    optimizations and caching can be done for efficiency.

    Reducing all subsets to "maximal" disjoint subsets amounts to
    pruning the list of subsets if the list of agent IDs is a strict subset
    of a larger list of agent IDs with overlap
    """
    agent_IDs = list(fovs.keys())
    fov_subsets = {
        frozenset({ID}): fov for ID, fov in fovs.items()
    }  # dict of agent ID sets to the subset
    overlap_ID_combos = {(ID,) for ID in agent_IDs}
    n = len(fovs)  # choose from all agents
    k = 2  # choose pairs of agents

    # Compute all disjoint subsets of overlap
    while True:
        # check all unique combinations of k agents for overlap
        found_a_combo = False
        agent_combos = itertools.combinations(range(n), k)
        for combo in agent_combos:
            IDs = frozenset([agent_IDs[idx] for idx in combo])

            # check if the km1 combos are valid
            all_subsets_valid = True
            all_km1_combos = list(itertools.combinations(IDs, k - 1))
            for km1_combo in all_km1_combos:
                if km1_combo not in overlap_ID_combos:
                    all_subsets_valid = False
                    break

            # check overlaps only if all subsets are valid
            if all_subsets_valid:
                # get starting point and remaining IDs
                IDs_use = all_km1_combos[0]  # just pick first arbitrarily
                overlap = fov_subsets[frozenset(IDs_use)]
                IDs_rem = IDs.difference(IDs_use)

                # check the remaining ID FOVs for overlap
                for ID in IDs_rem:
                    overlap = overlap.intersection(fovs[ID])
                    if overlap is None:
                        break
                else:
                    # modify the km1 subsets to reflect the disjoint nature
                    for km1_combo in all_km1_combos:
                        fov_subsets[frozenset(km1_combo)] = fov_subsets[
                            frozenset(km1_combo)
                        ].difference(overlap)
                    # add the new set to the dict
                    found_a_combo = True
                    fov_subsets[IDs] = overlap

        # check if we need to increment k and keep going
        if not found_a_combo:
            break  # we can't possibly find a more maximal one
        else:
            k += 1

    return fov_subsets


class Shape:
    pass


class Wedge(Shape):
    def __init__(self, radius: float, angle_start: float, angle_stop: float) -> None:
        """Define a wedge as a part of a circle

        Assumes unit circle convention where:
            (1, 0) is at 0
            (0, 1) is at pi/2
            (-1,0) is at pi or -pi
            (0,-1) is at -pi/2
        """
        self.radius = radius
        self.angle_range = [angle_start, angle_stop]

    def check_point(self, point: np.ndarray):
        """TODO: vectorize"""
        rng = np.linalg.norm(point)
        if rng <= self.radius:
            az = np.arctan2(point[1], point[0])  # between [-pi, pi]
            if self.angle_range[0] <= az <= self.angle_range[1]:
                return True
        return False


class Circle(Shape):
    def __init__(self, radius: float, center: np.ndarray) -> None:
        self.radius = radius
        self.center = center

    def intersection(self, other: "FieldOfView") -> Union["FieldOfView", None]:
        if isinstance(other, Circle):
            d = np.linalg.norm(self.center - other.center)
            if (d + self.radius) < other.radius:
                return self
            elif (d + other.radius) < self.radius:
                return other
            elif d < (self.radius + other.radius):
                overlap = Vesica([self, other])
            else:
                overlap = None
        elif isinstance(other, Vesica):
            overlap = other.intersection(self)
        else:
            raise NotImplementedError
        return overlap

    def difference(self, other: "FieldOfView") -> Union["FieldOfView", None]:
        raise NotImplementedError


class Vesica(Shape):
    def __init__(self, circles):
        """Shape formed by intersection of circles"""
        self.circles = circles

    def intersection(self, other: "FieldOfView") -> Union["FieldOfView", None]:
        if isinstance(other, Circle):
            for circle in self.circles:
                if np.linalg.norm(circle.center - other.center) >= max(
                    circle.radius, other.radius
                ):
                    overlap = None
                    break
            else:
                overlap = Vesica(self.circles + [other])
        elif isinstance(other, Vesica):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return overlap

    def difference(self, other: "FieldOfView") -> Union["FieldOfView", None]:
        raise NotImplementedError


class FieldOfView:
    def check_point(self, point: np.ndarray):
        raise NotImplementedError

    def intersection(self, other: "FieldOfView") -> Union["FieldOfView", None]:
        raise NotImplementedError

    def difference(self, other: "FieldOfView") -> Union["FieldOfView", None]:
        raise NotImplementedError


class ParametricFieldOfView(FieldOfView):
    def __init__(self) -> None:
        """Uses a particular shape to form a field of view"""
        super().__init__()

    def check_point(self, point: np.ndarray):
        return any([shape.check_point(point) for shape in self.shapes])


def fov_from_radar(range_doppler):
    raise NotImplementedError


def fov_from_lidar(pc: LidarData):
    """Use LiDAR data to estimate the field of view"""
    raise NotImplementedError
