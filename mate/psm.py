class PSM:
    def __init__(self, value, confidence):
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Pseudomeasurement: ({self.value}, {self.confidence})"
