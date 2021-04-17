from typing import Any, Optional, Union, Dict, List, Tuple
from copy import deepcopy
from sklearn.base import BaseEstimator, clone
import pandas as pd
from pandas.api.types import is_scalar


def isnull_allow_arrays(obj) -> bool:
    ret = False
    try:
        assert is_scalar(obj)
        ret = pd.isnull(obj)
    except Exception:
        pass
    return ret


class SavePredictMixin:
    def predict(self, X, **kwargs):
        if not hasattr(self, "_saved_preds"):
            self._saved_preds = {}
        r = super().predict(X, **kwargs)
        if not isnull_allow_arrays(r):
            self._saved_preds["predict"] = r
        return r


class SavePredictProbaMixin:
    def predict_proba(self, X, **kwargs):
        if not hasattr(self, "_saved_preds"):
            self._saved_preds = {}
        r = super().predict_proba(X, **kwargs)
        if not isnull_allow_arrays(r):
            self._saved_preds["predict_proba"] = r
        return r


class SaveDecisionFunctionMixin:
    def decision_function(self, X, **kwargs):
        if not hasattr(self, "_saved_preds"):
            self._saved_preds = {}
        r = super().decision_function(X, **kwargs)
        if not isnull_allow_arrays(r):
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
    try:
        obj = clone(obj)
    except Exception:
        obj = deepcopy(obj)
    obj.__class__ = subtype

    return obj
