from typing import Any, Optional, Union, Dict, List, Tuple

from sklearn.base import BaseEstimator, clone


class SavePredictMixin:
    def predict(self, X, **kwargs):
        if not hasattr(self, "_saved_preds"):
            self._saved_preds = {}
        r = super().predict(X, **kwargs)
        self._saved_preds["predict"] = r
        return r


class SavePredictProbaMixin:
    def predict_proba(self, X, **kwargs):
        if not hasattr(self, "_saved_preds"):
            self._saved_preds = {}
        r = super().predict_proba(X, **kwargs)
        self._saved_preds["predict_proba"] = r
        return r


class SaveDecisionFunctionMixin:
    def decision_function(self, X, **kwargs):
        if not hasattr(self, "_saved_preds"):
            self._saved_preds = {}
        r = super().decision_function(X, **kwargs)
        self._saved_preds["decision_function"] = r
        return r


def create_dynamically_subclassed_estimator(obj: BaseEstimator) -> BaseEstimator:
    assert isinstance(obj, BaseEstimator)
    subclasses = []
    if hasattr(obj, "predict"):
        subclasses.append(SavePredictMixin)
    if hasattr(obj, "predict_proba"):
        subclasses.append(SavePredictProbaMixin)
    if hasattr(obj, "decision_function"):
        subclasses.append(SaveDecisionFunctionMixin)
    return create_dynamically_subclassed_object(obj, subclasses)


def create_dynamically_subclassed_object(obj: Any, subclasses: List[type]) -> Any:
    original_type = type(obj)
    subtype = type(
        f"{original_type.__name__}{''.join([x.__name__ for x in subclasses])}",
        tuple(subclasses + [original_type]),
        {},
    )
    obj = clone(obj)
    obj.__class__ = subtype

    return obj
