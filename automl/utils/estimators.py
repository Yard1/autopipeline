from typing import Optional


class set_param_context:
    def __init__(self, estimator, cloned_estimators: Optional[list] = None, **params):
        self.estimator = estimator
        self.cloned_estimators = cloned_estimators
        self.params = params
        self.old_params = None

    def __enter__(self):
        self.old_params = {
            k: v for k, v in self.estimator.get_params().items() if k in self.params
        }
        self.estimator.set_params(**self.params)

    def __exit__(self, type, value, traceback):
        self.estimator.set_params(**self.old_params)
        if self.cloned_estimators:
            for cloned_estimator in self.cloned_estimators:
                try:
                    cloned_estimator.set_params(**self.old_params)
                except AttributeError:
                    pass
