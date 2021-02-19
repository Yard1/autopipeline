from enum import Enum, auto

class ProblemType(Enum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"

    def is_classification(self):
        return self in {self.BINARY, self.MULTICLASS}

    @staticmethod
    def translate(problem_type: str):
        if problem_type == "regression":
            return ProblemType.REGRESSION
        if problem_type in ("binary", "binary_classification"):
            return ProblemType.BINARY
        if problem_type in ("multiclass", "multilabel", "multiclass_classification", "multilabel_classification"):
            return ProblemType.MULTICLASS
        raise ValueError(f"Cannot translate '{problem_type}' to a ProblemType object!")