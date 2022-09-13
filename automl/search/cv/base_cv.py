from typing import Union
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    BaseShuffleSplit,
    BaseCrossValidator,
    StratifiedShuffleSplit,
    ShuffleSplit,
    GroupShuffleSplit,
)

from ...problems import ProblemType

SMALL_LARGE_THRES = 10000000
CV_HOLDOUT_THRESHOLD = 100000


def get_resampling_for_problem_type(
    problem_type: ProblemType,
    n_splits=5,
    groups=None,
    time_budget=None,
    shape=None,
    random_state=None,
) -> Union[BaseCrossValidator, BaseShuffleSplit]:
    if n_splits is None:
        if time_budget and isinstance(time_budget, list):
            time_budget = sum(time_budget)
        if (
            time_budget is None
            or shape[0] * shape[1] / 0.9 < SMALL_LARGE_THRES * (time_budget / 3600)
            and shape[0] < CV_HOLDOUT_THRESHOLD
        ):
            n_splits = 5
        else:
            n_splits = 0.1
    if not isinstance(n_splits, (int, float)):
        return n_splits
    if groups is not None:
        if isinstance(n_splits, int):
            return GroupKFold(n_splits=n_splits)
        return GroupShuffleSplit(
            n_splits=1, test_size=n_splits, random_state=random_state
        )
    if problem_type.is_classification():
        if isinstance(n_splits, int):
            return StratifiedKFold(n_splits=n_splits)
        return StratifiedShuffleSplit(
            n_splits=1, test_size=n_splits, random_state=random_state
        )
    if isinstance(n_splits, int):
        return KFold(n_splits=n_splits)
    return ShuffleSplit(n_splits=1, test_size=n_splits, random_state=random_state)


def get_cv_for_problem_type(
    problem_type: ProblemType, n_splits=5, groups=None
) -> BaseCrossValidator:
    if n_splits is None:
        n_splits = 5
    if not isinstance(n_splits, int):
        return n_splits
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    if problem_type.is_classification():
        return StratifiedKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits)
