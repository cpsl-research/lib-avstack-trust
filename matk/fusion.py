
class AggregatorFusion():
    """Simply appends all tracks together not worrying about duplicates"""

    def __call__(self, tracks_self: list, tracks_other: dict):
        tracks_out = tracks_self
        for tracks in tracks_other.values():
            tracks_out += tracks[1]
        return tracks_out