from typing import Any


class _Wrapper:
    def __init__(self, model, ID_local) -> None:
        self.model = model
        self.ID_local = ID_local

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class SensorWrapper(_Wrapper):
    def __init__(self, model, ID_local) -> None:
        super().__init__(model, ID_local)


class PerceptionWrapper(_Wrapper):
    def __init__(self, model, ID_local, sensor_ID_input) -> None:
        super().__init__(model, ID_local)
        self.sensor_ID_input = sensor_ID_input


class TrackingWrapper(_Wrapper):
    def __init__(self, model, ID_local, percep_ID_input) -> None:
        super().__init__(model, ID_local)
        self.percep_ID_input = percep_ID_input
