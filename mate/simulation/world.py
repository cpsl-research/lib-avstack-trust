from avstack.geometry import GlobalOrigin3D


class World:
    def __init__(self, dt, extent) -> None:
        self.objects = {}
        self.agents = {}
        self.tracks = {}
        self.frame = 0
        self.t = 0
        self.dt = dt
        self.extent = extent
        self.origin = GlobalOrigin3D

    def tick(self):
        self.frame += 1
        self.t += self.dt
        for obj in self.objects.values():
            obj.tick(self.dt)

    def add_object(self, obj):
        if obj.ID not in self.objects:
            self.objects[obj.ID] = obj

    def add_agent(self, agent):
        if agent.ID not in self.agents:
            self.agents[agent.ID] = agent
            self.tracks[agent.ID] = (None, [])

    def get_neighbors(self, agent_ID):
        """Get all agent neighbors of some agent"""
        neighbors = []
        for ID, other in self.agents.items():
            if agent_ID == -1:
                neighbors.append(ID)
            elif ID != self.agents[agent_ID]:
                if self.agents[agent_ID].in_range(other):
                    neighbors.append(ID)
        return neighbors

    def push_tracks(self, timestamp, agent_ID, tracks):
        """Receive all tracks from an agent"""
        self.tracks[agent_ID] = (timestamp, tracks)

    def pull_tracks(self, timestamp, agent_ID, with_timestamp=False):
        """Give all tracks to an agent if in range"""
        sends = {}
        for ID in self.get_neighbors(agent_ID):
            if with_timestamp:
                sends[ID] = self.tracks[ID]
            else:
                sends[ID] = self.tracks[ID][1]
        return sends
