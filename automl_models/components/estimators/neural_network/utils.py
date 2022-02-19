from typing import Dict, TYPE_CHECKING, List
import pandas as pd
from sklearn.base import is_classifier
from skorch.callbacks import EarlyStopping, GradientNormClipping, LRScheduler
import torch
import math


def get_category_cardinalities(
    category_cardinalities: Dict[str, int], X_cat: pd.DataFrame
) -> List[int]:
    if X_cat is None:
        return []
    if category_cardinalities:
        columns_set = set(X_cat.columns)
        category_cardinalities = [
            category_cardinalities[k]
            for k in sorted(
                {k for k in category_cardinalities if k in columns_set},
                key=list(X_cat.columns).index,
            )
        ]
    else:
        category_cardinalities = X_cat.nunique().to_list()
    return category_cardinalities


class AutoMLSkorchMixin:
    def get_default_callbacks(self):
        ret = super().get_default_callbacks()
        if self.verbose < 1:
            ret = [(name, callback) for name, callback in ret if name != "print_log"]
        return (
            ret
            + (
                [
                    (
                        "early_stopping",
                        EarlyStopping(
                            monitor="valid_loss", patience=self.n_iter_no_change
                        ),
                    )
                ]
                if self.early_stopping
                else []
            )
            + (
                [
                    (
                        "lr_scheduler",
                        LRScheduler(
                            torch.optim.lr_scheduler.OneCycleLR,
                            max_lr=self.scheduler_lr_,
                            step_every="batch",
                            pct_start=0.25,
                            final_div_factor=100000.0,
                            epochs=self.max_epochs,
                            steps_per_epoch=self.batches_per_epoch_,
                        ),
                    )
                ]
                if self.lr_schedule
                else []
            )
            + [
                (
                    "gradient_clipper",
                    GradientNormClipping(1.0),
                )
            ]
        )

    def fit(self, X, y=None, **kwargs):
        self.scheduler_lr_ = self.lr * 10
        self.batches_per_epoch_ = math.ceil(y.shape[0] / self.batch_size)
        return super().fit(X, y=y, **kwargs)

    def get_split_datasets(self, X, y=None, **fit_params):
        """Get internal train and validation datasets.

        The validation dataset can be None if ``self.train_split`` is
        set to None; then internal validation will be skipped.

        Override this if you want to change how the net splits
        incoming data into train and validation part.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        **fit_params : dict
          Additional parameters passed to the ``self.train_split``
          call.

        Returns
        -------
        dataset_train
          The initialized training dataset.

        dataset_valid
          The initialized validation dataset or None

        """
        dataset = self.get_dataset(X, y)
        if not self.train_split:
            return dataset, None

        if isinstance(self.train_split, type):
            if not hasattr(self, "train_split_"):
                self.train_split_ = self.train_split(
                    cv=self.cv,
                    random_state=self.random_state,
                    stratified=is_classifier(self),
                )
            train_split = self.train_split_
        else:
            train_split = self.train_split

        if y is None:
            return train_split(dataset, **fit_params)

        return train_split(dataset, y, **fit_params)

    def __joblib_hash__(self):
        if not hasattr(self, "module_"):
            return (
                self.__class__,
                {
                    k: v
                    for k, v in self.get_params().items()
                    if not k.startswith("callback")
                },
            )
        if hasattr(self, "fitted_dataset_hash_"):
            return (
                self.__class__,
                {
                    k: v
                    for k, v in self.get_params().items()
                    if not k.startswith("callback") and k not in ("_kwargs")
                },
                self.fitted_dataset_hash_,
            )
        return self

    def __joblib_hash_attrs_to_ignore__(self):
        return {k for k in self.__dict__ if k.startswith("callback")}
