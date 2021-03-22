from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.model_selection._validation import _score, _check_multimetric_scoring
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.metrics import make_scorer
import numpy as np
import gc
from copy import deepcopy
from collections import defaultdict

import ray
import ray.exceptions
from ray.tune import Trainable
import ray.cloudpickle as cpickle
import lz4.frame
from sklearn.utils.validation import check_is_fitted

from .utils import treat_config
from ..utils import f2_mcc_roc_auc
from ...utils.memory import dynamic_memory_factory
from ...utils.dynamic_subclassing import create_dynamically_subclassed_estimator


@ray.remote
class RayStore(object):
    @staticmethod
    def compress(value, **kwargs):
        if "compression_level" not in kwargs:
            kwargs["compression_level"] = 9
        return lz4.frame.compress(cpickle.dumps(value), **kwargs)

    @staticmethod
    def decompress(value):
        return cpickle.loads(lz4.frame.decompress(value))

    def __init__(self) -> None:
        self.values = defaultdict(dict)

    def get(self, key, store_name, pop=False, decompress=True):
        if pop:
            v = self.values[store_name].pop(key)
        else:
            v = self.values[store_name][key]
        gc.collect()
        if not decompress:
            return v
        return RayStore.decompress(v)

    def put(self, key, store_name, value):
        self.values[store_name][key] = RayStore.compress(value)

    def get_all_keys(self, store_name) -> list:
        return list(self.values[store_name].keys())

    def get_all_refs(self, store_name, pop=False) -> list:
        r = [self.get(key, store_name, pop=pop) for key in self.get_all_keys(store_name)]
        gc.collect()
        return r


class SklearnTrainable(Trainable):
    """Class to be passed in as the first argument of tune.run to train models.

    Overrides Ray Tune's Trainable class to specify the setup, train, save,
    and restore routines.

    """

    def setup(self, config, **params):
        # forward-compatbility
        self._setup(config, **params)

    def stop(self):
        super().stop()
        ray.actor.exit_actor()

    def _setup(self, config, **params):
        """Sets up Trainable attributes during initialization.

        Also sets up parameters for the sklearn estimator passed in.

        Args:
            config (dict): contains necessary parameters to complete the `fit`
                routine for the estimator. Also includes parameters for early
                stopping if it is set to true.

        """
        print("setup")
        if params:
            self.X_ = params["X_"]
            self.y_ = params["y_"]
            self.pipeline_blueprint = params["pipeline_blueprint"]
            self._component_strings_ = params["_component_strings_"]
            self.groups_ = params.get("groups_", None)
            self.fit_params = params.get("fit_params", None)
            self.scoring = params.get("scoring", None)
            self.metric_name = params.get("metric_name", "roc_auc")
            self.cv = deepcopy(params.get("cv", 5))
            self.n_jobs = params.get("n_jobs", None)
            self.random_state = deepcopy(params.get("random_state", None))
            self.prune_attr = params.get("prune_attr", None)
            self.const_values = params.get("const_values", {})
            self.cache = params.get("cache", None)
            self.X_test_ = params.get("X_test_", None)
            self.y_test_ = params.get("y_test_", None)
            self.cache_results = params.get("cache_results", True)
        assert self.X_ is not None
        self.estimator_config = config

    def step(self):
        # forward-compatbility
        print("training")
        return self._train()

    def _make_scoring_dict(self):
        scoring = self.scoring.copy()

        def dummy_score(y_true, y_pred):
            return 0

        dummy_pred_scorer = make_scorer(dummy_score, greater_is_better=True)
        dummy_pred_proba_scorer = make_scorer(
            dummy_score, greater_is_better=True, needs_proba=True
        )
        dummy_thereshold_scorer = make_scorer(
            dummy_score, greater_is_better=True, needs_threshold=True
        )
        scoring["dummy_pred_scorer"] = dummy_pred_scorer
        scoring["dummy_pred_proba_scorer"] = dummy_pred_proba_scorer
        scoring["dummy_thereshold_scorer"] = dummy_thereshold_scorer
        return scoring

    # TODO move this outside
    @staticmethod
    def score_test(
        estimator, X, y, X_test, y_test, scoring, refit=True, error_score=np.nan
    ):
        try:
            check_is_fitted(estimator)
        except:
            refit = True
        if refit:
            estimator = clone(estimator)
            estimator.fit(X, y)
        scores = _score(
            estimator,
            X_test,
            y_test,
            _check_multimetric_scoring(estimator, scoring),
            error_score=error_score,
        )
        scores["f2_mcc_roc_auc"] = f2_mcc_roc_auc(
            scores["matthews_corrcoef"], scores["roc_auc"]
        )
        return scores, estimator

    def _train(self):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config = {**self.const_values, **self.estimator_config}
        config_called = treat_config(
            config, self._component_strings_, self.random_state
        )
        if self.prune_attr:
            prune_attr = config_called.pop(self.prune_attr, None)
        else:
            prune_attr = None
        print(f"trial prune_attr: {prune_attr}")

        estimator.set_params(**config_called)
        memory = dynamic_memory_factory(self.cache)

        estimator.set_params(memory=memory)

        is_early_stopping_on = prune_attr and prune_attr < 1.0

        if is_early_stopping_on:
            subsample_cv = _SubsampleMetaSplitter(
                base_cv=self.cv,
                fraction=prune_attr,
                subsample_test=True,
                random_state=self.random_state,
            )
        else:
            subsample_cv = self.cv

        estimator_subclassed = create_dynamically_subclassed_estimator(estimator)
        scoring_with_dummies = self._make_scoring_dict()
        print(f"doing cv on {estimator_subclassed.steps[-1][1]}")
        scores = cross_validate(
            estimator_subclassed,
            self.X_,
            self.y_,
            cv=subsample_cv,
            groups=self.groups_,
            error_score="raise",
            return_estimator=True,
            verbose=0,
            scoring=scoring_with_dummies,
            fit_params=self.fit_params,
            # return_train_score=self.return_train_score,
        )
        print("cv done")

        estimator_fit_time = np.sum(
            [x.final_estimator_fit_time_ for x in scores["estimator"]]
        )
        metrics = {
            metric: np.mean(scores[f"test_{metric}"]) for metric in self.scoring.keys()
        }

        combined_predictions = {}
        if self.cache_results and not is_early_stopping_on:
            combined_predictions = {
                k: np.concatenate([x._saved_preds[k] for x in scores["estimator"]])
                for k in scores["estimator"][0]._saved_preds.keys()
            }

        gc.collect()

        ret = {}

        if self.cache_results:
            store = ray.get_actor("object_store")
        else:
            store = None

        test_metrics = None
        fitted_estimator = None
        if self.X_test_ is not None and not is_early_stopping_on:
            print("scoring test")
            test_metrics, fitted_estimator = SklearnTrainable.score_test(
                estimator, self.X_, self.y_, self.X_test_, self.y_test_, self.scoring
            )
            print("scoring test done")
            if store is not None:
                try:
                    store.put.remote(self.trial_id, "fitted_estimators", ray.put(fitted_estimator))
                except ray.exceptions.ObjectStoreFullError:
                    pass
                del fitted_estimator

        metrics["f2_mcc_roc_auc"] = f2_mcc_roc_auc(
            metrics["matthews_corrcoef"], metrics["roc_auc"]
        )

        metrics["f2_mcc_roc_auc"] = metrics["f2_mcc_roc_auc"] if metrics["f2_mcc_roc_auc"] else 0

        if self.metric_name:
            ret["mean_validation_score"] = np.mean(scores[f"test_{self.metric_name}"])
        else:
            ret["mean_validation_score"] = metrics["f2_mcc_roc_auc"]
        ret["estimator_fit_time"] = estimator_fit_time
        ret["metrics"] = metrics
        if test_metrics:
            ret["test_metrics"] = test_metrics
        if prune_attr:
            ret["dataset_fraction"] = prune_attr

        if store is not None:
            try:
                store.put.remote(self.trial_id, "fold_predictions", ray.put(combined_predictions))
            except ray.exceptions.ObjectStoreFullError:
                pass
            del combined_predictions

        gc.collect()
        print("done")
        return ret

    def reset_config(self, new_config):
        print("reset_config")
        self.estimator_config = new_config
        gc.collect()
        return True
