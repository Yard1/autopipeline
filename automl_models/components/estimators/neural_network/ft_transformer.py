from typing import Dict, Optional
from sklearn.base import clone
from skorch import NeuralNetClassifier, NeuralNetRegressor, NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping
from rtdl import FTTransformer as _FTTransformer, FeatureTokenizer, Transformer
import torch
import pandas as pd
import os

try:
    from automl.utils.memory.hashing import hash as xxd_hash
except ImportError:
    xxd_hash = None

from .utils import get_category_cardinalities, AutoMLSkorchMixin
from ...utils import validate_type


class FTTransformer(_FTTransformer):
    def forward(
        self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if x_cat.nelement() == 0:
            x_cat = None
        if x_num.nelement() == 0:
            x_num = None
        return super().forward(x_num, x_cat)


class FTTransformerClassifier(AutoMLSkorchMixin, NeuralNetClassifier):
    def __init__(
        self,
        module=FTTransformer,
        *,
        optimizer=torch.optim.AdamW,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=ValidSplit,
        classes=None,
        early_stopping: bool = True,
        random_state=None,
        category_cardinalities: Optional[Dict[str, set]] = None,
        n_iter_no_change=5,
        n_jobs=None,
        cv=0.2,
        **kwargs
    ):
        lr = kwargs.pop("lr", 1e-3)
        weight_decay = kwargs.pop("optimizer__weight_decay", 1e-5)
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer__weight_decay=weight_decay,
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
        self.category_cardinalities = category_cardinalities
        self.cv = cv

    @property
    def _default_callbacks(self):
        return super()._default_callbacks + (
            [
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=self.n_iter_no_change),
                )
            ]
            if self.early_stopping
            else []
        )

    def initialize_module(self):
        """Initializes the module.

        If the module is already initialized and no parameter was changed, it
        will be left as is.

        """
        kwargs = self.get_params_for("module")
        if "cat_cardinalities" in kwargs and not kwargs["cat_cardinalities"]:
            kwargs["cat_cardinalities"] = None
        module = self.module.make_default(**kwargs)
        module.__class__ = FTTransformer
        # pylint: disable=attribute-defined-outside-init
        self.module_ = module
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        validate_type(X, "X", pd.DataFrame)
        validate_type(y, "y", pd.Series)
        self.__dict__ = clone(self).__dict__
        if self.n_jobs and self.n_jobs > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
            torch.set_num_threads(self.n_jobs)
        self.train_split.random_state = self.random_state
        if self.random_state is not None:
            torch.random.manual_seed(self.random_state)
        X = X[sorted(X.columns)]
        if xxd_hash:
            self.fitted_dataset_hash_ = xxd_hash((X, y, fit_params))
        X_num = X.select_dtypes(exclude="category")
        X_cat = X.select_dtypes(include="category")
        category_cardinalities = get_category_cardinalities(
            self.category_cardinalities, X_cat
        )
        self.set_params(
            module__n_num_features=X_num.shape[1],
            module__cat_cardinalities=category_cardinalities,
            module__d_out=y.nunique(),
            module__last_layer_query_idx=[-1],
        )
        return super().fit(
            {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")},
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
                {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")}
            )
        return super().predict_proba(X)


class FTTransformerRegressor(AutoMLSkorchMixin, NeuralNetRegressor):
    def __init__(
        self,
        module=FTTransformer,
        *,
        optimizer=torch.optim.AdamW,
        criterion=torch.nn.MSELoss,
        train_split=ValidSplit,
        early_stopping: bool = True,
        random_state=None,
        category_cardinalities: Optional[Dict[str, set]] = None,
        n_iter_no_change=5,
        n_jobs=None,
        cv=0.2,
        **kwargs
    ):
        lr = kwargs.pop("lr", 1e-3)
        weight_decay = kwargs.pop("optimizer__weight_decay", 1e-5)
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer__weight_decay=weight_decay,
            lr=lr,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )

        self.early_stopping = early_stopping
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.category_cardinalities = category_cardinalities
        self.cv = cv

    @property
    def _default_callbacks(self):
        return super()._default_callbacks + (
            [
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=self.n_iter_no_change),
                )
            ]
            if self.early_stopping
            else []
        )

    def initialize_module(self):
        """Initializes the module.

        If the module is already initialized and no parameter was changed, it
        will be left as is.

        """
        kwargs = self.get_params_for("module")
        if "cat_cardinalities" in kwargs and not kwargs["cat_cardinalities"]:
            kwargs["cat_cardinalities"] = None
        module = self.module.make_default(**kwargs)
        module.__class__ = FTTransformer
        # pylint: disable=attribute-defined-outside-init
        self.module_ = module
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        validate_type(X, "X", pd.DataFrame)
        validate_type(y, "y", pd.Series)
        self.__dict__ = clone(self).__dict__
        if self.n_jobs and self.n_jobs > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
            torch.set_num_threads(self.n_jobs)
        self.train_split.random_state = self.random_state
        if self.random_state is not None:
            torch.random.manual_seed(self.random_state)
        X = X[sorted(X.columns)]
        if xxd_hash:
            self.fitted_dataset_hash_ = xxd_hash((X, y, fit_params))
        X_num = X.select_dtypes(exclude="category")
        X_cat = X.select_dtypes(include="category")
        category_cardinalities = get_category_cardinalities(
            self.category_cardinalities, X_cat
        )
        self.set_params(
            module__n_num_features=X_num.shape[1],
            module__cat_cardinalities=category_cardinalities,
            module__d_out=1,
            module__last_layer_query_idx=[-1],
        )
        return super().fit(
            {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")},
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
                {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")}
            )
        return super().predict_proba(X)
