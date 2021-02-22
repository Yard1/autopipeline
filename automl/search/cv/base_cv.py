from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    BaseCrossValidator,
)

from ...problems import ProblemType


def get_cv_for_problem_type(
    problem_type: ProblemType, n_splits=5, groups=None
) -> BaseCrossValidator:
    if n_splits is None:
        n_splits = 5
    if isinstance(n_splits, BaseCrossValidator):
        return n_splits
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    if problem_type.is_classification():
        return StratifiedKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits)