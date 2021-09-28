from automl.search.ensemble.ensemble_creator import EnsembleCreator
from automl.search.ensemble.stacking_ensemble_creator import StackingEnsembleCreator
import ray
import gc
import numpy as np
import joblib
from ray.util.joblib import register_ray
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
        ensemble.set_params(n_jobs=1)
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
    if hasattr(ensemble, "_saved_test_predictions"):
        del ensemble._saved_test_predictions
    try:
        ensemble.set_params(n_jobs=None)
    except Exception:
        pass
    print(scores)
    return scores


def cache_ensemble(
    ensemble,
    ensemble_name: str,
    current_stacking_level: int,
    ray_cache_actor: ray.ObjectRef,
):
    if not ray_cache_actor:
        return ensemble
    key = f"{current_stacking_level}_{ensemble_name}"
    ray.get(ray_cache_actor.put.remote(key, ENSEMBLE_STORE_NAME, ensemble))
    return ray.get(ray_cache_actor.get_cached_object.remote(key, ENSEMBLE_STORE_NAME))


@ray.remote(num_cpus=4, max_calls=1)
def ray_fit_ensemble_and_return_stacked_preds_remote(
    main_stacking_ensemble: StackingEnsembleCreator,
    ensemble_config,
    scoring_dict,
    ray_cache_actor: ray.ObjectRef,
):
    register_ray()
    with joblib.parallel_backend("sequential"):
        (
            main_stacking_ensemble_fitted,
            X_stack,
            X_test_stack,
        ) = main_stacking_ensemble.fit_ensemble_and_return_stacked_preds(
            **ensemble_config
        )
        scores = _score_ensemble(
            main_stacking_ensemble_fitted, ensemble_config, scoring_dict
        )
    main_stacking_ensemble_fitted = cache_ensemble(
        main_stacking_ensemble_fitted,
        main_stacking_ensemble._ensemble_name,
        ensemble_config["current_stacking_level"],
        ray_cache_actor,
    )
    gc.collect()
    return main_stacking_ensemble_fitted, X_stack, X_test_stack, scores


@ray.remote(num_cpus=4, max_calls=1)
def ray_fit_ensemble(
    ensemble: EnsembleCreator,
    ensemble_config,
    scoring_dict,
    ray_cache_actor: ray.ObjectRef,
):
    register_ray()
    with joblib.parallel_backend("sequential"):
        ensemble_fitted = ensemble.fit_ensemble(**ensemble_config)
        scores = _score_ensemble(ensemble_fitted, ensemble_config, scoring_dict)
    ensemble_fitted = cache_ensemble(
        ensemble_fitted,
        ensemble._ensemble_name,
        ensemble_config["current_stacking_level"],
        ray_cache_actor,
    )
    gc.collect()
    return ensemble_fitted, scores
