from enum import Enum, auto

class ProblemType(Enum):
    REGRESSION = auto()
    BINARY = auto()
    MULTICLASS = auto()

    def is_classification(self):
        return self in {self.BINARY, self.MULTICLASS}