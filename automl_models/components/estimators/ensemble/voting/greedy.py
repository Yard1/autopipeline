import math
import ray
from scipy.special import softmax
from copy import deepcopy
from functools import partial
from typing import List, Optional
import numpy as np
from collections import Counter

from sklearn.model_selection._validation import (
    _score,
    check_cv,
    is_classifier,
    _safe_indexing,
    check_scoring,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
)
from joblib import Parallel

from .voting import PandasVotingClassifier, PandasVotingRegressor
from ..utils import (
    get_ray_pg,
    get_cv_predictions,
    ray_pg_context,
    fit_estimators,
    clone_with_n_jobs,
    put_args_if_ray,
)


class FakeRegressor:
    def __init__(self, pred) -> None:
        self.pred = pred

    def predict(self, *args, **kwargs):
        return self.pred


class FakeClassifier(FakeRegressor):
    def predict(self, *args, **kwargs):
        try:
            return np.argmax(self.pred, axis=1)
        except np.AxisError:
            return self.pred

    def predict_proba(self, *args, **kwargs):
        return self.pred


def _get_initial_estimators(
    predictions: List[List[np.ndarray]],
    labels: List[np.ndarray],
    scorer,
    fake_estimator,
    n_initial_estimators,
):
    losses = [0] * len(predictions)
    for j, pred in enumerate(predictions):
        test_scores = [0] * len(predictions[0])
        for i in range(len(predictions[0])):
            test_scores[i] = _score(
                fake_estimator(pred[i]),
                None,
                labels[i],
                scorer,
                np.nan,
            )
        losses[j] = np.mean(test_scores)
    losses = list(enumerate(losses))
    losses.sort(key=lambda x: x[1], reverse=True)
    if isinstance(n_initial_estimators, float):
        n_initial_estimators = math.round(len(losses) * n_initial_estimators)
    n_initial_estimators = int(n_initial_estimators)
    return [i for i, _ in losses][:n_initial_estimators]


def _fit_greedy_ensemble(
    predictions: List[List[np.ndarray]],
    labels: List[np.ndarray],
    random_state,
    scorer,
    is_classifier_ensemble,
    n_initial_estimators,
    ensemble_size,
    n_splits,
    n_iter_no_change=None,
    actual_indices: Optional[List[int]] = None,
):
    rand: np.random.RandomState = check_random_state(random_state)
    fake_estimator = FakeClassifier if is_classifier_ensemble else FakeRegressor

    predictions = [ray.get(prediction) for prediction in predictions]

    ensemble: List[List[np.ndarray]] = []
    order = []
    test_score = -np.inf
    best_test_score = test_score

    if n_initial_estimators:
        order = _get_initial_estimators(
            predictions, labels, scorer, fake_estimator, n_initial_estimators
        )

    best_order = order
    iters_without_change = 0

    weighted_ensemble_predictions = [
        np.zeros(
            predictions[0][0].shape,
            dtype=np.float64,
        )
        for _ in range(n_splits)
    ]
    fant_ensemble_predictions = [
        np.zeros(
            weighted_ensemble_predictions[0].shape,
            dtype=np.float64,
        )
        for _ in range(n_splits)
    ]
    for i in range(ensemble_size):
        losses = [0] * len(predictions)
        s = len(ensemble)
        if s > 0:
            for i in range(len(weighted_ensemble_predictions)):
                np.add(
                    weighted_ensemble_predictions[i],
                    ensemble[-1][i],
                    out=weighted_ensemble_predictions[i],
                )

        # Memory-efficient averaging!
        for j, pred in enumerate(predictions):
            # fant_ensemble_prediction is the prediction of the current ensemble
            # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
            # We overwrite the contents of fant_ensemble_prediction
            # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
            test_scores = [0] * len(weighted_ensemble_predictions)
            for i in range(len(weighted_ensemble_predictions)):
                np.add(
                    weighted_ensemble_predictions[i],
                    pred[i],
                    out=fant_ensemble_predictions[i],
                )
                np.multiply(
                    fant_ensemble_predictions[i],
                    (1.0 / float(s + 1)),
                    out=fant_ensemble_predictions[i],
                )

                test_scores[i] = _score(
                    fake_estimator(fant_ensemble_predictions[i]),
                    None,
                    labels[i],
                    scorer,
                    np.nan,
                )

            # calculate_loss is versatile and can return a dict of losses
            # when scoring_functions=None, we know it will be a float
            # print(f"GREEDY TEST SCORES {test_scores}")
            losses[j] = np.mean(test_scores)

        all_best = np.argwhere(losses == np.nanmax(losses)).flatten()

        best = rand.choice(all_best)

        ensemble.append(predictions[best])
        order.append(best)
        test_score = losses[best]
        if test_score > best_test_score:
            best_test_score = test_score
            best_order = order
            iters_without_change = 0
        else:
            iters_without_change += 1
        print(
            f"GREEDY iter {i} max_iters {ensemble_size} train_score {test_score} best_train_score {best_test_score} iters_without_change {iters_without_change}"
        )

        # Handle special case
        if len(predictions) == 1:
            break

        if n_iter_no_change:
            if iters_without_change >= n_iter_no_change:
                break
        else:
            best_order = order

    if actual_indices:
        best_order = [actual_indices[i] for i in best_order]
    print(f"BAG ACTUAL INDICES {actual_indices} GREEDY INDICES {best_order}")
    return best_order


_ray_fit_greedy_ensemble = ray.remote(_fit_greedy_ensemble)


# TODO handle non-ray case
class GreedyEnsembleSelection:
    def _validate_weights(self):
        return

    def _fit_estimators(self, X, y, ests, sample_weight=None, groups=None):
        method = "predict"
        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )
        if getattr(self, "voting", "hard") == "soft":
            method = "predict_proba"
        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(
            parallel,
            self.n_jobs,
            len(ests),
            self.min_n_jobs_per_estimator,
            self.max_n_jobs_per_estimator,
        )
        X_ray, y_ray, sample_weight_ray = put_args_if_ray(parallel, X, y, sample_weight)
        with ray_pg_context(pg) as pg:
            predictions = get_cv_predictions(
                parallel=parallel,
                all_estimators=sorted(ests, key=lambda x: str(x)),
                X=X,
                y=y,
                X_ray=X_ray,
                y_ray=y_ray,
                cv=self.cv,
                fit_params=fit_params,
                verbose=self.verbose,
                stack_method=[method] * len(ests),
                n_jobs=self.n_jobs,
                pg=pg,
                return_type="refs",
            )

            cv = check_cv(deepcopy(self.cv), y, classifier=is_classifier(self))
            splits = list(cv.split(X, y, groups))
            labels = [_safe_indexing(y, test) for _, test in splits]

            self._bag_greedy_ensembles(predictions, labels, pg)
            self._calculate_weights()
            weights = self.weights_
            ests_to_fit = [
                ests[i] for i in range(len(ests)) if weights[i] and ests[i] != "drop"
            ]
            ests_fitted = fit_estimators(
                parallel,
                ests_to_fit,
                X,
                y,
                sample_weight_ray,
                partial(
                    clone_with_n_jobs,
                    n_jobs=int(pg.bundle_specs[-1]["CPU"]) if pg else 1,
                ),
                pg=pg,
                X_ray=X_ray,
                y_ray=y_ray,
            )
            self.estimators_ = []
            est_iter = iter(ests_fitted)
            for i in range(len(weights)):
                current_est = "drop" if not weights[i] else next(est_iter)
                self.estimators_.append(current_est)

    def _bag_greedy_ensembles(self, predictions, labels, pg):
        scorer = check_scoring(self, scoring=self.scoring)
        n_bags = self.n_bags or 1
        labels_ref = ray.put(labels)
        _ray_fit_greedy_ensemble_pg = _ray_fit_greedy_ensemble.options(
            placement_group=pg
        )
        # TODO handle non-ray case
        if n_bags == 1:
            self.indices_ = ray.get(
                _ray_fit_greedy_ensemble_pg.remote(
                    predictions,
                    labels_ref,
                    self.random_state,
                    scorer,
                    is_classifier(self),
                    self.n_initial_estimators,
                    self._ensemble_size_not_none,
                    self.cv.get_n_splits(),
                    self.n_iter_no_change,
                )
            )
        else:
            n_models = len(predictions)
            bag_size = int(n_models * self.bag_fraction)
            rand: np.random.RandomState = check_random_state(self.random_state)
            order_of_each_bag = []
            for j in range(n_bags):
                # Bagging a set of models
                indices = set(
                    rand.choice(np.arange(0, n_models), bag_size, replace=False)
                )
                if not indices:
                    continue
                bag_indices = [
                    (prediction, i)
                    for i, prediction in enumerate(predictions)
                    if i in indices
                ]
                bag, actual_indices = zip(*bag_indices)
                bag = list(bag)
                actual_indices = list(actual_indices)
                order = _ray_fit_greedy_ensemble_pg.remote(
                    bag,
                    labels_ref,
                    self.random_state + j,
                    scorer,
                    is_classifier(self),
                    self.n_initial_estimators,
                    self._ensemble_size_not_none,
                    self.cv.get_n_splits(),
                    self.n_iter_no_change,
                    actual_indices=actual_indices,
                )
                order_of_each_bag.append(order)
            if order_of_each_bag:
                order_of_each_bag = ray.get(order_of_each_bag)
                self.indices_ = [
                    item for sublist in order_of_each_bag for item in sublist
                ]
            else:
                self.indices_ = ray.get(
                    _ray_fit_greedy_ensemble_pg.remote(
                        predictions,
                        labels_ref,
                        self.random_state,
                        scorer,
                        is_classifier(self),
                        self.n_initial_estimators,
                        self._ensemble_size_not_none,
                        self.cv.get_n_splits(),
                        self.n_iter_no_change,
                    )
                )

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (len(self.estimators),),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weights[ensemble_member[0]] = float(ensemble_member[1])

        self.weights_ = softmax(weights)

        print(f"FINAL GREEDY INDICES {self.indices_} WEIGHTS {self.weights_}")

    @property
    def _ensemble_size_not_none(self) -> int:
        return self.ensemble_size or len(self.estimators)

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        check_is_fitted(self)
        if self.weights_ is None:
            return None
        return [
            w
            for est, w in zip(self.estimators, self.weights_)
            if est[1] != "drop" and w
        ]


class PandasGreedyVotingClassifier(GreedyEnsembleSelection, PandasVotingClassifier):
    def __init__(
        self,
        estimators,
        *,
        voting="soft",
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
        min_n_jobs_per_estimator=1,
        max_n_jobs_per_estimator=-1,
        n_initial_estimators=None,
        n_bags=20,
        bag_fraction=0.5,
        ensemble_size=None,
        n_iter_no_change=None,
        scoring=None,
        cv=None,
        random_state=None,
    ):
        self.estimators = estimators
        self.voting = voting
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        self.min_n_jobs_per_estimator = min_n_jobs_per_estimator
        self.max_n_jobs_per_estimator = max_n_jobs_per_estimator
        self.n_initial_estimators = n_initial_estimators
        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.n_iter_no_change = n_iter_no_change
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.ensemble_size = ensemble_size


class PandasGreedyVotingRegressor(GreedyEnsembleSelection, PandasVotingRegressor):
    def __init__(
        self,
        estimators,
        *,
        n_jobs=None,
        verbose=False,
        min_n_jobs_per_estimator=1,
        max_n_jobs_per_estimator=-1,
        n_initial_estimators=None,
        n_bags=20,
        bag_fraction=0.5,
        ensemble_size=None,
        n_iter_no_change=None,
        scoring=None,
        cv=None,
        random_state=None,
    ):
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.min_n_jobs_per_estimator = min_n_jobs_per_estimator
        self.max_n_jobs_per_estimator = max_n_jobs_per_estimator
        self.n_initial_estimators = n_initial_estimators
        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.cv = cv
        self.random_state = random_state
        self.ensemble_size = ensemble_size
