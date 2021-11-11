from typing import Optional
from skorch import NeuralNetClassifier, NeuralNetRegressor, NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping
from rtdl import FTTransformer as FTTransformer
import torch
import pandas as pd


class FTTransformerClassifier(NeuralNetClassifier):
    def __init__(
        self,
        module=FTTransformer,
        *,
        optimizer=torch.optim.AdamW,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=ValidSplit(0.2, stratified=True),
        classes=None,
        early_stopping: bool = True,
        random_state=None,
        n_iter_no_change=5,
        n_jobs=1,
        **kwargs
    ):
        lr = kwargs.pop("lr", 1e-4)
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
        module = self.module.make_default(**kwargs)
        # pylint: disable=attribute-defined-outside-init
        self.module_ = module
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        torch.set_num_threads(self.n_jobs)
        X = X[sorted(X.columns)]
        X_num = X.select_dtypes(exclude="category")
        X_cat = X.select_dtypes(include="category")
        self.set_params(
            module__n_num_features=X_num.shape[1],
            module__cat_cardinalities=X_cat.nunique().to_list(),
            module__d_out=y.nunique(),
            module__last_layer_query_idx=[-1],
        )
        self.train_split.random_state = self.random_state
        torch.random.manual_seed(self.random_state)
        return super().fit(
            {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")},
            y.to_numpy("int64"),
            **fit_params
        )

    def predict_proba(self, X):
        torch.set_num_threads(self.n_jobs)
        if isinstance(X, pd.DataFrame):
            X = X[sorted(X.columns)]
            X_num = X.select_dtypes(exclude="category")
            X_cat = X.select_dtypes(include="category")
            return super().predict_proba(
                {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")}
            )
        return super().predict_proba(X)


class FTTransformerRegressor(NeuralNetRegressor):
    def __init__(
        self,
        module=FTTransformer,
        *,
        optimizer=torch.optim.AdamW,
        criterion=torch.nn.MSELoss,
        train_split=ValidSplit(0.2, stratified=False),
        early_stopping: bool = True,
        random_state=None,
        n_iter_no_change=5,
        n_jobs = 1,
        **kwargs
    ):
        lr = kwargs.pop("lr", 1e-4)
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
        module = self.module.make_default(**kwargs)
        # pylint: disable=attribute-defined-outside-init
        self.module_ = module
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        torch.set_num_threads(self.n_jobs)
        X = X[sorted(X.columns)]
        X_num = X.select_dtypes(exclude="category")
        X_cat = X.select_dtypes(include="category")
        self.set_params(
            module__n_num_features=X_num.shape[1],
            module__cat_cardinalities=X_cat.nunique().to_list(),
            module__d_out=1,
            module__last_layer_query_idx=[-1],
        )
        self.train_split.random_state = self.random_state
        torch.random.manual_seed(self.random_state)
        return super().fit(
            {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")},
            y.to_numpy("int64"),
            **fit_params
        )

    def predict(self, X):
        torch.set_num_threads(self.n_jobs)
        if isinstance(X, pd.DataFrame):
            X = X[sorted(X.columns)]
            X_num = X.select_dtypes(exclude="category")
            X_cat = X.select_dtypes(include="category")
            return super().predict(
                {"x_num": X_num.to_numpy("float32"), "x_cat": X_cat.to_numpy("int32")}
            )
        return super().predict(X)
