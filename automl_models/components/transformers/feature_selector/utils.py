import numpy as np
import shap
import contextlib
import warnings

try:
    import fasttreeshap
except ImportError:
    fasttreeshap = None

# lightgbm_rf_config = {
#     "n_jobs": 1,
#     "boosting_type": "rf",
#     "max_depth": 5,
#     "num_leaves": 32,
#     "subsample": 0.632,
#     "subsample_freq": 1,
#     "verbose": -1,
#     "learning_rate": 0.05,
#     "class_weight": "balanced",
# }

lightgbm_fs_config = {
    "n_jobs": 1,
    "max_depth": 5,
    "num_leaves": 32,
    "class_weight": "balanced",
    "verbose": -1,
}


def get_shap(estimator, X, n_jobs=1) -> np.ndarray:
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        try:
            assert fasttreeshap
            explainer = fasttreeshap.TreeExplainer(
                estimator, feature_perturbation="tree_path_dependent", n_jobs=n_jobs
            )
            shap_values = np.array(explainer.shap_values(X))
        except Exception:
            explainer = shap.TreeExplainer(
                estimator, feature_perturbation="tree_path_dependent"
            )
            shap_values = np.array(explainer.shap_values(X))
        if len(shap_values.shape) == 3:
            shap_values = np.abs(shap_values).sum(axis=0)
            shap_values = shap_values.mean(0)
        else:
            shap_values = np.abs(shap_values).mean(0)
        return shap_values


def get_tree_num(estimator, n_feat: int) -> int:
    depth = None
    try:
        depth = estimator.get_params()["max_depth"]
    except KeyError:
        warnings.warn(
            "The estimator does not have a max_depth property, as a result "
            " the number of trees to use cannot be estimated automatically."
        )
    if depth is None:
        depth = 10
    # how many times a feature should be considered on average
    f_repr = 100
    # n_feat * 2 because the training matrix is extended with n shadow features
    multi = (n_feat * 2) / (np.sqrt(n_feat * 2) * depth)
    n_estimators = int(multi * f_repr)
    return n_estimators
