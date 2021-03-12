from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
import numpy as np
import os
from pickle import PicklingError
import warnings
import inspect
from copy import deepcopy
import time

import ray
from ray.tune import Trainable
import ray.cloudpickle as cpickle

from .utils import treat_config


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
            self.groups_ = params.pop("groups_", None)
            self.fit_params = params.pop("fit_params", None)
            self.scoring = params.pop("scoring", None)
            self.metric_name = params.pop("metric_name", "roc_auc")
            self.cv = deepcopy(params.pop("cv", 5))
            self.n_jobs = params.pop("n_jobs", None)
            self.random_state = params.pop("random_state", None)
            self.prune_attr = params.pop("prune_attr", None)
            self.const_values = params.pop("const_values", {})
        assert self.X_ is not None
        self.estimator_config = config

    def step(self):
        # forward-compatbility
        print("training")
        return self._train()

    def _train(self):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config = {**self.const_values, **self.estimator_config}
        config_called = treat_config(config, self._component_strings_, self.random_state)
        if self.prune_attr:
            prune_attr = config_called.pop(self.prune_attr, None)
        else:
            prune_attr = None
        print(f"trial prune_attr: {prune_attr}")

        estimator.set_params(**config_called)
        # memory = dynamic_memory_factory(self._cache)
        # estimator.set_params(memory=memory)

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
            # fit_params=self.fit_params,
            # groups=self.groups,
            # return_train_score=self.return_train_score,
        )

        estimator_fit_time = np.sum(
            [x.final_estimator_fit_time_ for x in scores["estimator"]]
        )
        metrics = {
            metric: np.mean(scores[f"test_{metric}"]) for metric in self.scoring.keys()
        }

        ret = {}

        ret["mean_test_score"] = np.mean(scores[f"test_{self.metric_name}"])
        ret["estimator_fit_time"] = estimator_fit_time
        ret["metrics"] = metrics
        if prune_attr:
            ret["dataset_fraction"] = prune_attr

        return ret

    def reset_config(self, new_config):
        print("reset_config")
        self.estimator_config = new_config
        return True