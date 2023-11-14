from avstack.config import MODELS
from avstack.geometry import GlobalOrigin3D


@MODELS.register_module()
class World:
    def __init__(self, dt, dimensions, extent) -> None:
        self._object_map = {}
        self._agent_map = {}
        self.tracks = {}
        self.frame = 0
        self.t = 0
        self.dt = dt
        self.dimensions = dimensions
        self.extent = extent
        self.reference = GlobalOrigin3D

    @property
    def objects(self):
        return list(self._object_map.values())

    @property
    def agents(self):
        return list(self._agent_map.values())

    def tick(self):
        self.frame += 1
        self.t += self.dt
        for obj in self.objects:
            obj.tick(self.dt)

    def add_object(self, obj):
        if obj.ID not in self._object_map:
            self._object_map[obj.ID] = obj

    def add_agent(self, agent):
        if agent.ID not in self._agent_map:
            self._agent_map[agent.ID] = agent
            self.tracks[agent.ID] = (None, {})

    def get_neighbors(self, agent_ID):
        """Get all agent neighbors of some agent"""
        neighbors = []
        for ID, other in self._agent_map.items():
            if agent_ID == -1:
                neighbors.append(ID)
            elif ID != self._agent_map[agent_ID]:
                if self._agent_map[agent_ID].in_range(other):
                    neighbors.append(ID)
        return neighbors

    def push_tracks(self, timestamp, agent_ID, tracks):
        """Receive all tracks from an agent"""
        self.tracks[agent_ID] = (timestamp, tracks)

    def pull_tracks(self, timestamp, with_timestamp=False):
        """Give all tracks to an agent if in range"""
        assert not with_timestamp
        sends = {
            ID: (self.tracks[ID][1][0] if len(self.tracks[ID][1]) > 0 else [])
            for ID in self.tracks
        }  # HACK for only 1 tracker per agent
        return sends
