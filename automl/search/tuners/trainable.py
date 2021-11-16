import os
import numpy as np
import gc
import sys
import time
from copy import deepcopy
from ray.tune.utils.placement_groups import PlacementGroupFactory
from ray.tune.resources import Resources
from ray.util.placement_group import get_current_placement_group
from sklearn.base import clone
import joblib

from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.utils import resample, _safe_indexing

import ray
import ray.exceptions

from ray.tune import Trainable

from ray.util.joblib import register_ray

from .utils import treat_config
from ..utils import ray_score_test, ray_cross_validate, stack_estimator
from ..metrics.scorers import make_scorer_with_error_score
from ..metrics.metrics import optimized_precision
from ...problems.problem_type import ProblemType
from ...utils.memory import dynamic_memory_factory
from ...utils.estimators import set_param_context
from ...utils.tune_callbacks import META_KEY

import logging

logger = logging.getLogger(__name__)


class _SubsampleMetaSplitterWithStratify(_SubsampleMetaSplitter):
    """Splitter that subsamples a given fraction of the dataset"""

    def __init__(
        self, *, base_cv, fraction, subsample_test, random_state, stratify=False
    ):
        super().__init__(
            base_cv=base_cv,
            fraction=fraction,
            subsample_test=subsample_test,
            random_state=random_state,
        )
        self.stratify = stratify

    def split(self, X, y, groups=None):
        for train_idx, test_idx in self.base_cv.split(X, y, groups):
            train_idx = resample(
                train_idx,
                replace=False,
                random_state=self.random_state,
                n_samples=int(self.fraction * train_idx.shape[0]),
                stratify=_safe_indexing(y, train_idx) if self.stratify else None,
            )
            if self.subsample_test:
                test_idx = resample(
                    test_idx,
                    replace=False,
                    random_state=self.random_state,
                    n_samples=int(self.fraction * test_idx.shape[0]),
                    stratify=_safe_indexing(y, test_idx) if self.stratify else None,
                )
            yield train_idx, test_idx





# TODO break this up into a class for classification and for regression
# TODO save all split scores
class SklearnTrainable(Trainable):
    """Class to be passed in as the first argument of tune.run to train models.

    Overrides Ray Tune's Trainable class to specify the setup, train, save,
    and restore routines.

    """

    N_JOBS = 1

    def setup(self, config, refs, **params):
        # forward-compatbility
        self.refs = refs[0]
        self._setup(config, **params)

    def _setup(self, config, **params):
        """Sets up Trainable attributes during initialization.

        Also sets up parameters for the sklearn estimator passed in.

        Args:
            config (dict): contains necessary parameters to complete the `fit`
                routine for the estimator. Also includes parameters for early
                stopping if it is set to true.

        """
        if params:
            self.X_ = params["X_"]
            self.y_ = params["y_"]
            self.pipeline_blueprint = params["pipeline_blueprint"]
            self.component_strings = params["component_strings"]
            self.hyperparameter_names = params["hyperparameter_names"]
            self.problem_type = params["problem_type"]
            self.groups_ = params.get("groups_", None)
            self.fit_params = params.get("fit_params", None)
            self.scoring = params.get("scoring", None)
            self.metric_name = params["metric_name"]
            self.cv = params.get("cv", 5)
            self.random_state = params.get("random_state", None)
            self.prune_attr = params.get("prune_attr", None)
            self.const_values = params.get("const_values", {})
            self.cache = params.get("cache", None)
            self.X_test_ = params.get("X_test_", None)
            self.y_test_ = params.get("y_test_", None)
            self.cache_results = params.get("cache_results", True)
            self.previous_stack = params.get("previous_stack", None)
            self.N_JOBS = params.get("n_jobs", 1)
            self.n_jobs_per_fold = params.get("n_jobs_per_fold", 1)
        assert self.X_ is not None
        self.estimator_config = config
        self.estimator = None

    def step(self):

        # forward-compatbility
        logger.debug("training")
        register_ray()
        with joblib.parallel_backend("threading"):
            r = self._train()

        return r

    def _make_scoring_dict(self):
        scoring = self.scoring.copy()

        def dummy_score(y_true, y_pred):
            return 0

        dummy_pred_scorer = make_scorer_with_error_score(
            dummy_score, greater_is_better=True, error_score=0
        )
        scoring["dummy_pred_scorer"] = dummy_pred_scorer
        if self.problem_type.is_classification():
            dummy_pred_proba_scorer = make_scorer_with_error_score(
                dummy_score, greater_is_better=True, needs_proba=True, error_score=0
            )
            scoring["dummy_pred_proba_scorer"] = dummy_pred_proba_scorer
            dummy_decision_function_scorer = make_scorer_with_error_score(
                dummy_score, greater_is_better=True, needs_threshold=True, error_score=0
            )
            scoring["dummy_decision_function_scorer"] = dummy_decision_function_scorer
        return scoring

    #@classmethod
    #def default_resource_request(cls, config):
        #return Resources(cpu=0, gpu=0, extra_cpu=cls.N_JOBS, extra_gpu=0)

    def _train(self):
        self.estimator = None
        time_cv = time.time()
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config = {**self.const_values, **self.estimator_config}
        config.pop(META_KEY, None)
        config_called = treat_config(
            config, self.component_strings, self.hyperparameter_names, self.random_state
        )
        if self.prune_attr:
            prune_attr = config_called.pop(self.prune_attr, None)
        else:
            prune_attr = None
        #print(f"trial prune_attr: {prune_attr}")

        estimator.set_params(**config_called)
        memory = dynamic_memory_factory(self.cache)

        estimator.set_params(memory=memory)
        #print(f"doing cv on {estimator.steps[-1][1]}")

        n_jobs_per_fold = self.n_jobs_per_fold

        estimator.set_params(
            **{
                k: n_jobs_per_fold
                for k, v in estimator.get_params().items()
                if k.endswith("n_jobs") or k.endswith("thread_count")
            }
        )

        if self.previous_stack:
            self.previous_stack.set_params(n_jobs=n_jobs_per_fold)
            estimator = stack_estimator(estimator, self.previous_stack)

        is_early_stopping_on = prune_attr and prune_attr < 1.0

        if is_early_stopping_on:
            subsample_cv = _SubsampleMetaSplitterWithStratify(
                base_cv=self.cv,
                fraction=prune_attr,
                subsample_test=True,
                random_state=self.random_state,
                stratify=self.problem_type.is_classification(),
            )
        else:
            subsample_cv = self.cv

        # TODO: threshold for binary classification? https://github.com/scikit-learn/scikit-learn/pull/16525/files
        # TODO: prediction time (per row)

        scoring_with_dummies = self._make_scoring_dict()
        # print(self.X_.columns)

        # TODO do this better - we want the prefix to be dynamic
        prefix = "<class 'automl.search.tuners.tuner.SklearnTrainable'>_"

        # print({
        #     k: v
        #     for k, v in estimator.get_params().items()
        #     if k.endswith("n_jobs") or k.endswith("thread_count")
        # })

        test_ret = None
        if self.X_test_ is not None and not is_early_stopping_on:
            print("scoring test")
            # print({
            #     k: v
            #     for k, v in estimator.get_params().items()
            #     if k.endswith("n_jobs") or k.endswith("thread_count")
            # })
            test_ret = ray_score_test.options(num_cpus=n_jobs_per_fold, placement_group=get_current_placement_group()).remote(
                estimator,
                self.X_,
                self.y_,
                self.X_test_,
                self.y_test_,
                scoring_with_dummies,
            )

        scores = ray_cross_validate(
            estimator,
            self.X_,
            self.y_,
            cv=subsample_cv,
            groups=self.groups_,
            error_score="raise",
            return_estimator=False,
            verbose=1,
            scoring=scoring_with_dummies,
            fit_params=self.fit_params,
            n_jobs=n_jobs_per_fold,
            X_ref=self.refs[prefix + "X_"],
            y_ref=self.refs[prefix + "y_"]
            # return_train_score=self.return_train_score,
        )
        print("cv done")

        estimator_fit_time = time.time() - time_cv
        metrics = {
            metric: np.mean(scores[f"test_{metric}"]) for metric in self.scoring.keys()
        }

        if self.problem_type == ProblemType.BINARY:
            metrics["optimized_precision"] = optimized_precision(
                metrics["accuracy"], metrics["recall"], metrics["specificity"]
            )

        del scores

        test_metrics = None

        if test_ret:
            test_metrics, fitted_estimator = ray.get(test_ret)
            self.estimator = fitted_estimator
            test_metrics = {
                k: v for k, v in test_metrics.items() if k in self.scoring
            }
            if self.problem_type == ProblemType.BINARY:
                test_metrics["optimized_precision"] = optimized_precision(
                    test_metrics["accuracy"],
                    test_metrics["recall"],
                    test_metrics["specificity"],
                )
            logger.debug("scoring test done")
        else:
            self.estimator = estimator

        ret = {}

        ret["mean_validation_score"] = metrics[self.metric_name]
        ret["estimator_fit_time"] = estimator_fit_time
        ret["metrics"] = metrics
        if test_metrics:
            ret["test_metrics"] = test_metrics
        if prune_attr:
            ret["dataset_fraction"] = prune_attr

        ret["done"] = True
        print("done")
        #ret["size"] = sys.getsizeof(fitted_estimator)
        return ret

    def reset_config(self, new_config):
        print("reset_config")
        self.estimator_config = new_config
        gc.collect()
        return True

    def save_checkpoint(self, tmp_checkpoint_dir):
        joblib.dump(self.estimator, os.path.join(tmp_checkpoint_dir, "pipeline.pkl"))
        return tmp_checkpoint_dir