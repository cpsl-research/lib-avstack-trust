from avstack.datastructs import PriorityQueue


class World:
    def __init__(self, dt) -> None:
        self.objects = {}
        self.agents = {}
        self.tracks = {}
        self.t = 0
        self.dt = dt

    def tick(self):
        self.t += self.dt
        for obj in self.objects.values():
            obj.tick()

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
            if ID != self.agents[agent_ID]:
                if self.agents[agent_ID].in_range(other):
                    neighbors.append(ID)
        return neighbors

    def receive_tracks(self, timestamp, agent_ID, tracks):
        """Receive all tracks from an agent"""
        self.tracks[agent_ID] = (timestamp, tracks)

    def send_tracks(self, timestamp, agent_ID):
        """Send all tracks to an agent if in range"""
        sends = {}
        for ID in self.get_neighbors(agent_ID):
            sends[ID] = self.tracks[ID]
        return sends
