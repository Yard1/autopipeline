from sklearn.base import clone


def clone_with_n_jobs_1(estimator, *, safe=True):
    estimator = clone(estimator, safe=safe)
    params = estimator.get_params()
    params_to_set = {param: 1 for param in params.keys() if param.endswith("n_jobs")}
    estimator.set_params(**params_to_set)
    # clone twice to deal with nested
    estimator = clone(estimator, safe=safe)
    return estimator
