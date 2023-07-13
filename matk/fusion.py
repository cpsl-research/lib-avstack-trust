from copy import deepcopy
from avstack.modules.fusion import ci_fusion
from avstack.modules.tracking.tracks import XyFromRazTrack


class AggregatorFusion():
    """Simply appends all tracks together not worrying about duplicates"""
    def __call__(self, tracks_self: list, tracks_other: dict):
        tracks_out = tracks_self
        for tracks in tracks_other.values():
            tracks_out += tracks[1]
        return tracks_out
    

class CovarianceIntersectionFusion():
    """Runs assignment algorithm to determine if there are duplicates
    
    For duplicates, run covariance intersection for fusion
    """
    def __init__(self, clustering):
        self.clustering = clustering

    def __call__(self, tracks_self: list, tracks_other: dict):
        # -- run clustering
        objects = [tracks_self.data]
        objects.extend([tracks[1].data if len(tracks[1]) > 0 else [] for 
                        tracks in tracks_other.values()])
        clusters = self.clustering(objects)

        # -- run fusion
        tracks_out = []
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                # perform fusion on the array
                x_fuse, P_fuse = cluster[0].x, cluster[0].P
                for track in cluster[1:]:
                    x_fuse, P_fuse = ci_fusion(
                        x_fuse, P_fuse, track.x, track.P, w=0.5
                    )
                # rebuild the track
                track = XyFromRazTrack(
                    t0=cluster[0].t0,
                    raz=None,
                    reference=cluster[0].reference,
                    obj_type=cluster[0].obj_type,
                    ID_force=i,
                    x=x_fuse,
                    P=P_fuse,
                    t=cluster[0].t,
                    coast=-1,
                    n_updates=-1,
                    age=-1
                )
                tracks_out.append(track)

        return tracks_out