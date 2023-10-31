import numpy as np
from avstack.modules.fusion import clustering


class BasicObject:
    def __init__(self, x, ID) -> None:
        self.x = x
        self.ID = ID

    @property
    def position(self):
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Object {self.ID}"

    def distance(self, other):
        return np.linalg.norm(self.x - other.x)


def make_objects_idx_list(n_clustered=10, n_objects=20, n_sets=5):
    n_lone = n_objects - n_clustered
    n_per_cluster = np.random.randint(low=2, high=n_sets, size=n_clustered)
    idx_set_cluster = [
        np.random.choice(n_sets, size=n, replace=False) for n in n_per_cluster
    ]
    idx_set_lone = np.random.choice(n_sets, size=n_objects - n_clustered)

    objects_idx_list = []
    for i_set in range(n_sets):
        this_lone = [i_obj for i_obj, idx in enumerate(idx_set_lone) if idx == i_set]
        this_clust = [
            i_obj + n_lone
            for i_obj, idxs in enumerate(idx_set_cluster)
            for idx in idxs
            if idx == i_set
        ]
        objects_idx_list.append(this_lone + this_clust)

    return objects_idx_list, idx_set_lone, idx_set_cluster


def make_objects(n_objects, sigma=100, n_states=6):
    return [BasicObject(sigma * np.random.randn(n_states), i) for i in range(n_objects)]


def test_make_objects_list_1():
    objects_idx_list, idx_set_lone, idx_set_cluster = make_objects_idx_list(
        n_clustered=0, n_objects=10, n_sets=5
    )
    assert len({x for l in objects_idx_list for x in l}) == 10
    assert sum([len(lst) for lst in objects_idx_list]) == 10
    assert sum([len(lst) for lst in idx_set_cluster]) == 0
    assert len(idx_set_lone) == 10


def test_make_objects_list_2():
    objects_idx_list, idx_set_lone, idx_set_cluster = make_objects_idx_list(
        n_clustered=10, n_objects=10, n_sets=5
    )
    assert len({x for l in objects_idx_list for x in l}) == 10
    assert sum([len(lst) for lst in objects_idx_list]) > 10
    assert sum([len(lst) for lst in idx_set_cluster]) > 0
    assert len(idx_set_lone) == 0


def test_sampled_clustering():
    # Make clustering observations
    n_objects = 20
    n_clusters = 10
    n_sets = 5
    objects = make_objects(n_objects=n_objects)
    objects_idx_list, idx_set_lone, idx_set_cluster = make_objects_idx_list(
        n_clustered=n_clusters, n_objects=n_objects, n_sets=n_sets
    )
    observations = [[objects[i] for i in obj_list] for obj_list in objects_idx_list]

    # Run clustering algorithm
    clusterer = clustering.SampledAssignmentClustering(assign_radius=1)
    clusters = clusterer(observations)
    assert len(clusters) == n_objects
    assert len([c for c in clusters if len(c) > 1]) == n_clusters
