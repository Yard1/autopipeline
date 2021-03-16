from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.model_selection._validation import _score, _check_multimetric_scoring
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
import numpy as np
import gc
from pickle import PicklingError
import warnings
import inspect
from copy import deepcopy
import time

import ray
from ray.tune import Trainable
import ray.cloudpickle as cpickle

from .utils import treat_config
from ...utils.memory import dynamic_memory_factory


class SklearnTrainable(Trainable):
    """Class to be passed in as the first argument of tune.run to train models.

    Overrides Ray Tune's Trainable class to specify the setup, train, save,
    and restore routines.

    """

    def setup(self, config, **params):
        # forward-compatbility
        self._setup(config, **params)

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
            self.fit_params = params.get("fit_params", {})
        assert self.X_ is not None
        self.estimator_config = config

    def step(self):
        # forward-compatbility
        print("training")
        return self._train()

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

        if prune_attr and prune_attr < 1.0:
            subsample_cv = _SubsampleMetaSplitter(
                base_cv=self.cv,
                fraction=prune_attr,
                subsample_test=True,
                random_state=self.random_state,
            )
        else:
            subsample_cv = self.cv

        scores = cross_validate(
            estimator,
            self.X_,
            self.y_,
            cv=subsample_cv,
            groups=self.groups_,
            error_score="raise",
            return_estimator=True,
            verbose=0,
            scoring=self.scoring,
            fit_params=self.fit_params,
            # return_train_score=self.return_train_score,
        )

        estimator_fit_time = np.sum(
            [x.final_estimator_fit_time_ for x in scores["estimator"]]
        )
        metrics = {
            metric: np.mean(scores[f"test_{metric}"]) for metric in self.scoring.keys()
        }

        gc.collect()

        ret = {}

        test_metrics = None
        if self.X_test_ is not None and not (prune_attr and prune_attr < 1.0):
            test_estimator = clone(estimator)
            test_estimator.fit(self.X_, self.y_)
            test_metrics = _score(
                test_estimator,
                self.X_test_,
                self.y_test_,
                _check_multimetric_scoring(test_estimator, self.scoring),
                error_score=np.nan,
            )

        # ret["mean_validation_score"] = np.mean(scores[f"test_{self.metric_name}"])
        ret["mean_validation_score"] = self._mean_validation_score(
            metrics["matthews_corrcoef"], metrics["roc_auc"]
        )
        ret["estimator_fit_time"] = estimator_fit_time
        ret["metrics"] = metrics
        if test_metrics:
            ret["test_metrics"] = test_metrics
        if prune_attr:
            ret["dataset_fraction"] = prune_attr

        return ret

    def reset_config(self, new_config):
        print("reset_config")
        self.estimator_config = new_config
        gc.collect()
        return True

    # TODO move this outside
    def _mean_validation_score(self, mcc, roc_auc, beta=2):
        roc_auc = (roc_auc * 2) - 1
        return (1 + beta) * (mcc * roc_auc) / ((beta * mcc) + roc_auc)
