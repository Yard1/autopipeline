from typing import Optional
from sklearn.base import clone


def clone_with_n_jobs_1(estimator, *, safe=True):
    estimator = clone(estimator, safe=safe)
    params = estimator.get_params()
    params_to_set = {param: 1 for param in params.keys() if param.endswith("n_jobs")}
    estimator.set_params(**params_to_set)
    # clone twice to deal with nested
    estimator = clone(estimator, safe=safe)
    return estimator


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
                except Exception:
                    pass
