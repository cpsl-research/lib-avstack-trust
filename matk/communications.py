

class CommsModel():
    def __init__(self, max_range) -> None:
        self.max_range = max_range

    def in_range(self):
        raise NotImplementedError


class Omnidirectional(CommsModel):
    def __init__(self, max_range=100) -> None:
        super().__init__(max_range=max_range)

    def in_range(self, agent_1, agent_2):
        return agent_1.position.distance(agent_2.position) < self.max_range