from automl.search.ensemble.ensemble_creator import EnsembleCreator
from automl.search.ensemble.stacking_ensemble_creator import StackingEnsembleCreator
import numpy as np
import traceback
from ..utils import score_test

ENSEMBLE_STORE_NAME = "ensembles"


def _score_ensemble(
    ensemble,
    ensemble_config,
    scoring_dict,
):
    X = ensemble_config["X"]
    y = ensemble_config["y"]
    X_test = ensemble_config.get("X_test_original", None)
    y_test = ensemble_config.get("y_test_original", None)
    try:
        ensemble.set_params(n_jobs=-1)
    except Exception:
        pass
    if X_test is None:
        scores = None
    else:
        scores, _ = score_test(
            ensemble,
            X,
            y,
            X_test,
            y_test,
            scoring_dict,
            refit=False,
            error_score=np.nan,
        )
    try:
        ensemble.set_params(n_jobs=None)
    except Exception:
        pass
    print(scores)
    return scores


# @ray.remote(num_cpus=4, max_calls=1)
def ray_fit_ensemble_and_return_stacked_preds_remote(
    main_stacking_ensemble: StackingEnsembleCreator,
    ensemble_config,
    scoring_dict,
):
    # with joblib.parallel_backend("sequential"):
    main_stacking_ensemble_fitted = (
        main_stacking_ensemble.fit_ensemble_and_return_stacked_preds(**ensemble_config)
    )
    scores = _score_ensemble(
        main_stacking_ensemble_fitted, ensemble_config, scoring_dict
    )
    return main_stacking_ensemble_fitted, scores


# @ray.remote(num_cpus=4, max_calls=1)
def ray_fit_ensemble(
    ensemble: EnsembleCreator,
    ensemble_config,
    scoring_dict,
):
    # with joblib.parallel_backend("sequential"):
    print(f"ray_fit_ensemble {ensemble}")
    try:
        ensemble_fitted = ensemble.fit_ensemble(**ensemble_config)
        scores = _score_ensemble(ensemble_fitted, ensemble_config, scoring_dict)
    except Exception:
        traceback.print_exc()
        ensemble_fitted, scores = None, None
    return ensemble_fitted, scores
