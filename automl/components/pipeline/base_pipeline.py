from sklearn.pipeline import Pipeline

class BasePipeline(Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False, column_types=None):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.column_types = column_types
        self._validate_steps()