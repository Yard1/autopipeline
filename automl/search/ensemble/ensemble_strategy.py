from typing import Any, Dict, List
import pandas as pd
import numpy as np
from abc import ABC
import os
from collections import ChainMap, defaultdict
from itertools import cycle, islice

import logging

logger = logging.getLogger(__name__)
DELIM = os.environ["TUNE_RESULT_DELIM"]


def _merge_dicts(dicts: List[dict]) -> dict:
    if not isinstance(dicts, list):
        dicts = [dicts]
    return dict(ChainMap(*dicts))


def _discard_below_max_dataset_fraction(results: Dict[str, Any]):
    max_dataset_fraction = max(
        (
            result["dataset_fraction"]
            for result in results.values()
            if "dataset_fraction" in result
        ),
        default=-np.inf,
    )
    return {
        trial_id: result
        for trial_id, result in results.items()
        if result.get("dataset_fraction", 0) >= max_dataset_fraction
    }


def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)  # .next on Python 2
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


class EnsembleStrategy(ABC):
    def __init__(
        self,
        configurations_to_select: int,
        percentile_threshold: int,
        use_only_last_results: bool = True,
    ) -> None:
        self.configurations_to_select = configurations_to_select
        self.percentile_threshold = percentile_threshold
        self.use_only_last_results = use_only_last_results
        super().__init__()

    def select_trial_ids(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: List[Dict[str, Any]],
        pipeline_blueprint,
    ) -> list:
        return None

    def select_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.use_only_last_results:
            results = results[-1]
        return _discard_below_max_dataset_fraction(_merge_dicts(results))

    def get_percentile(self, results: Dict[str, Any]):
        return np.percentile(
            [result["mean_validation_score"] for result in results.values()],
            self.percentile_threshold,
        )


class RoundRobinEstimator(EnsembleStrategy):
    def select_trial_ids(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: List[Dict[str, Any]],
        pipeline_blueprint,
    ) -> list:
        results = self.select_results(results)
        if self.configurations_to_select < 0:
            return set(results)
        percentile = self.get_percentile(results)
        results_per_estimator = defaultdict(list)
        for trial_id, result in results.items():
            if "config" not in result or "Estimator" not in result["config"]:
                continue
            results_per_estimator[
                f"{result['config']['Estimator'].split('(')[0]}_{result['config']['meta']['stacking_level']}"
            ].append(result)
        for estimator in results_per_estimator:
            results_per_estimator[estimator] = sorted(
                [
                    result
                    for result in results_per_estimator[estimator]
                    if result.get("mean_validation_score", -np.inf) >= percentile
                ],
                key=lambda x: x.get("mean_validation_score", -np.inf),
                reverse=True,
            )
        results_per_estimator = list([r for r in results_per_estimator.values() if r])
        results_per_estimator = sorted(
            results_per_estimator,
            key=lambda x: x[0].get("mean_validation_score", -np.inf),
            reverse=True,
        )
        selected_trials_ids = [
            result["trial_id"] for result in _roundrobin(*results_per_estimator)
        ][: self.configurations_to_select]
        return selected_trials_ids


class EnsembleBest(EnsembleStrategy):
    def select_trial_ids(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: List[Dict[str, Any]],
        pipeline_blueprint,
    ) -> list:
        results = self.select_results(results)
        if self.configurations_to_select < 0:
            return set(results)
        percentile = self.get_percentile(results)
        sorted_results = sorted(
            [
                result
                for trial_id, result in results.items()
                if result.get("mean_validation_score", -np.inf) >= percentile
            ],
            key=lambda x: x.get("mean_validation_score", -np.inf),
            reverse=True,
        )
        return [
            result["trial_id"]
            for result in sorted_results[: self.configurations_to_select]
        ]


class OneRoundRobinThenEnsembleBest(EnsembleStrategy):
    def select_trial_ids(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: List[Dict[str, Any]],
        pipeline_blueprint,
    ) -> list:
        return None
