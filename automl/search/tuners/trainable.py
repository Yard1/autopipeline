import numpy as np
import gc
import time
from copy import deepcopy
from collections import defaultdict
from sklearn.base import clone

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.metrics import make_scorer
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

import joblib

from ray.tune import Trainable
from ray.tune.resources import Resources
import ray.cloudpickle as cpickle

import lz4.frame

from ...utils.joblib_backend import register_ray_caching
from .utils import treat_config, split_list_into_chunks
from ..utils import score_test
from ..metrics.scorers import make_scorer_with_error_score
from ..metrics.metrics import optimized_precision
from ...problems.problem_type import ProblemType
from ...utils.types import array_shrink
from ...utils.memory import dynamic_memory_factory
from ...utils.dynamic_subclassing import create_dynamically_subclassed_estimator
from ...utils.estimators import set_param_context

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


def compress(value, **kwargs):
    if "compression_level" not in kwargs:
        kwargs["compression_level"] = 9
    return lz4.frame.compress(cpickle.dumps(value), **kwargs)


def decompress(value):
    return cpickle.loads(lz4.frame.decompress(value))


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
        if not decompress:
            return v
        return RayStore.decompress(v)

    def put(self, key, store_name, value, compress=True):
        self.values[store_name][key] = RayStore.compress(value) if compress else value
        if compress:
            del value

    def get_all_keys(self, store_name) -> list:
        return list(self.values[store_name].keys())

    def get_all_refs(self, store_name, pop=False) -> list:
        r = [
            self.get(key, store_name, pop=pop) for key in self.get_all_keys(store_name)
        ]
        return r


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
            self._component_strings_ = params["_component_strings_"]
            self.problem_type = params["problem_type"]
            self.groups_ = params.get("groups_", None)
            self.fit_params = params.get("fit_params", None)
            self.scoring = params.get("scoring", None)
            self.metric_name = params["metric_name"]
            self.cv = deepcopy(params.get("cv", 5))
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
        logger.debug("training")
        register_ray_caching()
        r = self._train()

        gc.collect()
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
        return Resources(
            # cpu=cls.N_JOBS,
            cpu=1,
            gpu=0,
            extra_cpu=cls.N_JOBS,
        )

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
        for future in results_futures:
            del future
        del results_futures

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

    def _combine_cv_repeat(self, chunk):
        predictions, test_indices = zip(*chunk)
        predictions = np.concatenate(predictions)
        test_indices = np.concatenate(test_indices)
        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))
        return predictions[inv_test_indices]

    def _combine_cv_predictions(self, predictions, train_test_indices):
        if isinstance(self.cv, _RepeatedSplits):
            repeats = self.cv.n_repeats
        else:
            repeats = 1
        _, test_indices = zip(*train_test_indices)
        predictions_with_indices = list(zip(predictions, test_indices))
        combined_predictions = [
            array_shrink(self._combine_cv_repeat(repeat), int2uint=True)
            for repeat in split_list_into_chunks(
                predictions_with_indices, len(predictions_with_indices) // repeats
            )
        ]
        return combined_predictions

    def _train(self):
        time_cv = time.time()
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config = {**self.const_values, **self.estimator_config}
        config_called = treat_config(
            config, self._component_strings_, self.random_state
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

        if self.cache_results and not is_early_stopping_on:
            (
                estimator_subclassed,
                original_type,
            ) = create_dynamically_subclassed_estimator(estimator)
        else:
            estimator_subclassed, original_type = estimator, None

        scoring_with_dummies = self._make_scoring_dict()
        logger.debug(f"doing cv on {estimator_subclassed.steps[-1][1]}")
        # with joblib.parallel_backend("ray_caching"):
        scores = self._cross_validate(
            estimator_subclassed,
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

        combined_predictions = {}
        if self.cache_results and not is_early_stopping_on:
            combined_predictions = {
                k: self._combine_cv_predictions(
                    [x._saved_preds[k] for x in scores["estimator"]],
                    scores["cv_indices"],
                )
                for k in scores["estimator"][0]._saved_preds.keys()
            }

        if self.problem_type == ProblemType.BINARY:
            metrics["optimized_precision"] = optimized_precision(
                metrics["accuracy"], metrics["recall"], metrics["specificity"]
            )

        del scores

        gc.collect()

        ret = {}

        test_metrics = None
        fitted_estimator = None
        fitted_estimator_list = []
        combined_test_predictions = {}

        if self.X_test_ is not None and not is_early_stopping_on:
            logger.debug("scoring test")
            with set_param_context(
                estimator_subclassed,
                cloned_estimators=fitted_estimator_list,
                # TODO: look into why setting n_jobs to >1 here leads to way slower results
                **{
                    k: 1
                    for k, v in estimator_subclassed.get_params().items()
                    if k.endswith("n_jobs")
                },
            ):
                test_ret = ray_score_test.remote(
                    estimator_subclassed,
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
                if self.cache_results:
                    combined_test_predictions = {
                        k: array_shrink(
                            fitted_estimator._saved_preds[k],
                            int2uint=True,
                        )
                        for k in fitted_estimator._saved_preds.keys()
                    }
                fitted_estimator_list.append(fitted_estimator)
            fitted_estimator = fitted_estimator_list[0]
            if original_type is not None:
                del fitted_estimator._saved_preds
                fitted_estimator.__class__ = original_type
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

        if self.cache_results:
            try:
                store = ray.get_actor("object_store")
            except ValueError as e:
                logger.warning(e)
                store = None
        else:
            store = None

        if store is not None:
            try:
                store.put.remote(
                    self.trial_id,
                    "fold_predictions",
                    compress(combined_predictions),
                    False,
                )
            except ray.exceptions.ObjectStoreFullError:
                pass
            if fitted_estimator is not None:
                try:
                    store.put.remote(
                        self.trial_id,
                        "fitted_estimators",
                        compress(fitted_estimator),
                        False,
                    )
                except ray.exceptions.ObjectStoreFullError:
                    pass
            if combined_test_predictions:
                try:
                    store.put.remote(
                        self.trial_id,
                        "test_predictions",
                        compress(combined_test_predictions),
                        False,
                    )
                except ray.exceptions.ObjectStoreFullError:
                    pass

        del combined_predictions
        del fitted_estimator
        del combined_test_predictions

        logger.debug("done")
        return ret

    def reset_config(self, new_config):
        logger.debug("reset_config")
        self.estimator_config = new_config
        gc.collect()
        return True
