from typing import Dict, Optional
from sklearn.base import clone
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.dataset import ValidSplit
from fastai.tabular.model import TabularModel as _TabularModel, emb_sz_rule
import torch
import pandas as pd
import os

try:
    from automl.utils.memory.hashing import hash as xxd_hash
except ImportError:
    xxd_hash = None

from .utils import get_category_cardinalities, AutoMLSkorchMixin
from ...utils import validate_type


def _one_emb_sz(n_cat, n):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = {}
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat, sz


def get_emb_sz(sizes: list, columns: list):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"

    return tuple(_one_emb_sz(size, column) for size, column in zip(sizes, columns))


class TabularModel(_TabularModel):
    def __init__(
        self,
        emb_szs,
        n_cont,
        out_sz,
        layers,
        ps=None,
        embed_p=0.0,
        y_range=None,
        use_bn=True,
        bn_final=False,
        bn_cont=True,
        act_cls=torch.nn.ReLU(inplace=True),
        lin_first=True,
    ):
        layers = list(layers)
        emb_szs = list(emb_szs)
        return super().__init__(
            emb_szs,
            n_cont,
            out_sz,
            layers,
            ps=ps,
            embed_p=embed_p,
            y_range=y_range,
            use_bn=use_bn,
            bn_final=bn_final,
            bn_cont=bn_cont,
            act_cls=act_cls,
            lin_first=lin_first,
        )


class FastAINNClassifier(AutoMLSkorchMixin, NeuralNetClassifier):
    def __init__(
        self,
        module=TabularModel,
        *,
        optimizer=torch.optim.AdamW,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=ValidSplit,
        classes=None,
        batch_size_power=None,
        early_stopping: bool = True,
        random_state=None,
        category_cardinalities: Optional[Dict[str, set]] = None,
        n_iter_no_change=5,
        n_jobs=None,
        cv=0.2,
        lr_schedule=True,
        **kwargs
    ):
        lr = kwargs.pop("lr", 1e-3)
        layers = kwargs.pop("module__layers", (200, 100))
        super().__init__(
            module=module,
            module__layers=layers,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
            train_split=train_split,
            classes=classes,
            **kwargs
        )
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.batch_size_power = batch_size_power
        self.category_cardinalities = category_cardinalities
        self.cv = cv
        self.lr_schedule = lr_schedule

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        validate_type(X, "X", pd.DataFrame)
        validate_type(y, "y", pd.Series)
        self.__dict__ = clone(self).__dict__
        if self.n_jobs and self.n_jobs > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
            torch.set_num_threads(self.n_jobs)
        if self.random_state is not None:
            torch.random.manual_seed(self.random_state)
        if self.batch_size_power:
            self.set_params(batch_size=2 ** self.batch_size_power)
        X = X[sorted(X.columns)]
        if xxd_hash:
            self.fitted_dataset_hash_ = xxd_hash((X, y, fit_params))
        X_num = X.select_dtypes(exclude="category")
        X_cat = X.select_dtypes(include="category")
        self.category_cardinalities_ = get_category_cardinalities(
            self.category_cardinalities, X_cat
        )
        emb_szs = get_emb_sz(self.category_cardinalities_, list(X_cat.columns))
        self.set_params(
            module__emb_szs=emb_szs,
            module__n_cont=X_num.shape[1],
            module__out_sz=y.nunique(),
        )
        return super().fit(
            {"x_cont": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")},
            y.to_numpy("int64"),
            **fit_params
        )

    def predict_proba(self, X):
        if self.n_jobs and self.n_jobs > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
            torch.set_num_threads(self.n_jobs)
        if isinstance(X, pd.DataFrame):
            X = X[sorted(X.columns)]
            X_num = X.select_dtypes(exclude="category")
            X_cat = X.select_dtypes(include="category")
            return super().predict_proba(
                {"x_cont": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")}
            )
        return super().predict_proba(X)


class FastAINNRegressor(AutoMLSkorchMixin, NeuralNetRegressor):
    def __init__(
        self,
        module=TabularModel,
        *,
        optimizer=torch.optim.AdamW,
        criterion=torch.nn.MSELoss,
        train_split=ValidSplit,
        batch_size_power=None,
        early_stopping: bool = True,
        random_state=None,
        category_cardinalities: Optional[Dict[str, set]] = None,
        n_iter_no_change=5,
        n_jobs=None,
        cv=0.2,
        lr_schedule=True,
        **kwargs
    ):
        lr = kwargs.pop("lr", 1e-3)
        layers = kwargs.pop("module__layers", (200, 100))

        super().__init__(
            module=module,
            module__layers=layers,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )

        self.early_stopping = early_stopping
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.batch_size_power = batch_size_power
        self.category_cardinalities = category_cardinalities
        self.cv = cv
        self.lr_schedule = lr_schedule

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        validate_type(X, "X", pd.DataFrame)
        validate_type(y, "y", pd.Series)
        self.__dict__ = clone(self).__dict__
        if self.n_jobs and self.n_jobs > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
            torch.set_num_threads(self.n_jobs)
        if self.random_state is not None:
            torch.random.manual_seed(self.random_state)
        if self.batch_size_power:
            self.set_params(batch_size=2 ** self.batch_size_power)
        X = X[sorted(X.columns)]
        if xxd_hash:
            self.fitted_dataset_hash_ = xxd_hash((X, y, fit_params))
        X_num = X.select_dtypes(exclude="category")
        X_cat = X.select_dtypes(include="category")
        category_cardinalities = get_category_cardinalities(
            self.category_cardinalities, X_cat
        )
        emb_szs = get_emb_sz(category_cardinalities, list(X_cat.columns))
        self.set_params(
            module__emb_szs=emb_szs,
            module__n_cont=X_num.shape[1],
            module__out_sz=1,
        )
        return super().fit(
            {"x_cont": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")},
            y.to_numpy("int64"),
            **fit_params
        )

    def predict(self, X):
        if self.n_jobs and self.n_jobs > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
            torch.set_num_threads(self.n_jobs)
        if isinstance(X, pd.DataFrame):
            X = X[sorted(X.columns)]
            X_num = X.select_dtypes(exclude="category")
            X_cat = X.select_dtypes(include="category")
            return super().predict(
                {"x_cont": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")}
            )
        return super().predict(X)
