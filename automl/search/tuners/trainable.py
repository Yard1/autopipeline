import numpy as np
import gc
import sys
import time
from copy import deepcopy
from ray.tune.utils.placement_groups import PlacementGroupFactory
from ray.tune.resources import Resources
from sklearn.base import clone

from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.model_selection._validation import (
    indexable,
    check_cv,
    is_classifier,
    check_scoring,
    _check_multimetric_scoring,
    _insert_error_scores,
    _normalize_score_results,
    _aggregate_score_dicts,
    _fit_and_score,
)
from sklearn.utils import resample, _safe_indexing
from sklearn.model_selection._split import _RepeatedSplits

import ray
import ray.exceptions

from ray.tune import Trainable

from ray.util.joblib import register_ray

from .utils import treat_config, split_list_into_chunks
from ..utils import score_test
from ..metrics.scorers import make_scorer_with_error_score
from ..metrics.metrics import optimized_precision
from ...problems.problem_type import ProblemType
from ...utils.types import array_shrink
from ...utils.memory import dynamic_memory_factory
from ...utils.dynamic_subclassing import create_dynamically_subclassed_estimator
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


@ray.remote
def ray_fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    return _fit_and_score(
        estimator=estimator,
        X=X,
        y=y,
        scorer=scorer,
        train=train,
        test=test,
        verbose=verbose,
        parameters=parameters,
        fit_params=fit_params,
        return_train_score=return_train_score,
        return_parameters=return_parameters,
        return_n_test_samples=return_n_test_samples,
        return_times=return_times,
        return_estimator=return_estimator,
        split_progress=split_progress,
        candidate_progress=candidate_progress,
        error_score=error_score,
    )


@ray.remote(num_returns=2)
def ray_score_test(
    estimator,
    X,
    y,
    X_test,
    y_test,
    scoring,
    refit: bool = True,
    error_score=np.nan,
):
    return score_test(
        estimator=estimator,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        scoring=scoring,
        refit=refit,
        error_score=error_score,
    )


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
        logger.debug("setup")
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
        assert self.X_ is not None
        self.estimator_config = config

    def step(self):

        # forward-compatbility
        logger.debug("training")
        register_ray()
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

    @classmethod
    def default_resource_request(cls, config):
        return Resources(cpu=0, gpu=0, extra_cpu=cls.N_JOBS, extra_gpu=0)

    def _cross_validate(
        self,
        estimator,
        X,
        y=None,
        *,
        groups=None,
        scoring=None,
        cv=None,
        n_jobs=None,
        verbose=0,
        fit_params=None,
        pre_dispatch="2*n_jobs",
        return_train_score=False,
        return_estimator=False,
        error_score=np.nan,
    ):
        """Fast cross validation with Ray, adapted from sklearn.validation.cross_validate"""
        X, y, groups = indexable(X, y, groups)

        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        if callable(scoring):
            scorers = scoring
        elif scoring is None or isinstance(scoring, str):
            scorers = check_scoring(estimator, scoring)
        else:
            scorers = _check_multimetric_scoring(estimator, scoring)

        # TODO do this better - we want the prefix to be dynamic
        prefix = "<class 'automl.search.tuners.tuner.SklearnTrainable'>_"

        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        train_test = list(cv.split(X, y, groups))

        results_futures = [
            ray_fit_and_score.remote(
                clone(estimator),
                self.refs[prefix + "X_"],
                self.refs[prefix + "y_"],
                scorers,
                train,
                test,
                verbose,
                None,
                fit_params,
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            for train, test in train_test
        ]

        results = ray.get(results_futures)

        # For callabe scoring, the return type is only know after calling. If the
        # return type is a dictionary, the error scores can now be inserted with
        # the correct key.
        if callable(scoring):
            _insert_error_scores(results, error_score)

        results = _aggregate_score_dicts(results)

        ret = {}
        ret["fit_time"] = results["fit_time"]
        ret["score_time"] = results["score_time"]

        if return_estimator:
            ret["estimator"] = results["estimator"]

        test_scores_dict = _normalize_score_results(results["test_scores"])
        if return_train_score:
            train_scores_dict = _normalize_score_results(results["train_scores"])

        for name in test_scores_dict:
            ret["test_%s" % name] = test_scores_dict[name]
            if return_train_score:
                key = "train_%s" % name
                ret[key] = train_scores_dict[name]

        # added in automl
        ret["cv_indices"] = train_test

        return ret

    def _train(self):
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
        logger.debug(f"trial prune_attr: {prune_attr}")

        estimator.set_params(**config_called)
        memory = dynamic_memory_factory(self.cache)

        estimator.set_params(memory=memory)

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
        logger.debug(f"doing cv on {estimator.steps[-1][1]}")
        # print(self.X_.columns)
        scores = self._cross_validate(
            estimator,
            self.X_,
            self.y_,
            cv=subsample_cv,
            groups=self.groups_,
            error_score="raise",
            return_estimator=True,
            verbose=1,
            scoring=scoring_with_dummies,
            fit_params=self.fit_params,
            n_jobs=self.N_JOBS,
            # return_train_score=self.return_train_score,
        )
        logger.debug("cv done")

        estimator_fit_time = time.time() - time_cv
        metrics = {
            metric: np.mean(scores[f"test_{metric}"]) for metric in self.scoring.keys()
        }

        if self.problem_type == ProblemType.BINARY:
            metrics["optimized_precision"] = optimized_precision(
                metrics["accuracy"], metrics["recall"], metrics["specificity"]
            )

        del scores

        ret = {}

        test_metrics = None
        fitted_estimator = None
        fitted_estimator_list = []

        if self.X_test_ is not None and not is_early_stopping_on:
            logger.debug("scoring test")
            with set_param_context(
                estimator,
                cloned_estimators=fitted_estimator_list,
                # TODO: look into why setting n_jobs to >1 here leads to way slower results
                **{
                    k: 1
                    for k, v in estimator.get_params().items()
                    if k.endswith("n_jobs")
                },
            ):
                test_ret = ray_score_test.remote(
                    estimator,
                    self.X_,
                    self.y_,
                    self.X_test_,
                    self.y_test_,
                    scoring_with_dummies,
                )
                test_metrics, fitted_estimator = ray.get(test_ret)
                test_metrics = {
                    k: v for k, v in test_metrics.items() if k in self.scoring
                }
                fitted_estimator_list.append(fitted_estimator)
            fitted_estimator = fitted_estimator_list[0]
            if self.problem_type == ProblemType.BINARY:
                test_metrics["optimized_precision"] = optimized_precision(
                    test_metrics["accuracy"],
                    test_metrics["recall"],
                    test_metrics["specificity"],
                )
            logger.debug("scoring test done")

        ret["mean_validation_score"] = metrics[self.metric_name]
        ret["estimator_fit_time"] = estimator_fit_time
        ret["metrics"] = metrics
        if test_metrics:
            ret["test_metrics"] = test_metrics
        if prune_attr:
            ret["dataset_fraction"] = prune_attr

        logger.debug("done")
        ret["size"] = sys.getsizeof(fitted_estimator)
        return ret

    def reset_config(self, new_config):
        logger.debug("reset_config")
        self.estimator_config = new_config
        gc.collect()
        return True
