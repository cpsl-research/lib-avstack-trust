from avstack.config import MODELS


class CommsModel:
    def __init__(self, rate, max_range, send, receive) -> None:
        self.rate = rate
        self.max_range = max_range
        self.send = send
        self.receive = receive

    def in_range(self):
        raise NotImplementedError


@MODELS.register_module()
class Omnidirectional(CommsModel):
    def __init__(
        self, rate="continuous", max_range=100, send=True, receive=True
    ) -> None:
        super().__init__(rate=rate, max_range=max_range, send=send, receive=receive)

    def in_range(self, agent_1, agent_2):
        return (
            agent_1.position.distance(agent_2.position) < self.max_range
            if self.max_range
            else True
        )
