from avstack.config import MODELS
from avstack.geometry import GlobalOrigin3D


@MODELS.register_module()
class World:
    def __init__(self, dt, dimensions, extent) -> None:
        self._object_map = {}
        self._agent_map = {}
        self.agent_data = {}
        self.frame = 0
        self.timestamp = 0
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
        self.timestamp += self.dt
        for obj in self.objects:
            obj.tick(self.dt)

    def add_object(self, obj):
        if obj.ID not in self._object_map:
            self._object_map[obj.ID] = obj

    def add_agent(self, agent):
        if agent.ID not in self._agent_map:
            self._agent_map[agent.ID] = agent
            self.agent_data[agent.ID] = (None, {})

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

    def push_agent_data(self, timestamp, agent_ID, data):
        """Receive all data from an agent"""
        self.agent_data[agent_ID] = (timestamp, data)

    def pull_agent_data(self, timestamp, with_timestamp=False, target_reference=None):
        """Give all data to an agent if in range"""
        assert not with_timestamp
        sends = {
            ID: self.agent_data[ID][1] for ID in self.agent_data
        }  # HACK for only 1 tracker per agent
        if target_reference:
            for ID, send in sends.items():
                if (send) and (len(send) > 0):
                    send.apply("change_reference", target_reference, inplace=False)
        return sends
