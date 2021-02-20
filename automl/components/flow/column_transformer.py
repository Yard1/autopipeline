from copy import deepcopy
from sklearn.compose import ColumnTransformer as _ColumnTransformer

from .flow import Flow
from .utils import (
    recursively_remove_invalid_components,
    get_single_component_from_iterable,
    is_component_valid_iterable,
    get_step_choice_grid,
)
from ..component import ComponentLevel, ComponentConfig
from ...search.stage import AutoMLStage

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.utils.validation import check_array


class PandasColumnTransformer(_ColumnTransformer):
    def _hstack(self, Xs):
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
        """
        Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
        try:
            if all(isinstance(X, (pd.DataFrame, pd.Series)) for X in Xs):
                return pd.concat(Xs, axis=1)
        except:
            pass
        return np.hstack(Xs)


class ColumnTransformer(Flow):
    _component_class = PandasColumnTransformer
    _default_parameters = {
        "remainder": "passthrough",
        "sparse_threshold": 0,
        "n_jobs": None,
        "transformer_weights": None,
        "verbose": False,
    }

    @property
    def components_name(self) -> str:
        return "transformers"

    def __call__(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ):
        params = deepcopy(self.final_parameters)
        transformers = [
            (
                name,
                get_single_component_from_iterable(
                    transformer,
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                )(pipeline_config=pipeline_config, current_stage=current_stage),
                columns,
            )
            for name, transformer, columns in params["transformers"]
            if is_component_valid_iterable(
                transformer,
                pipeline_config=pipeline_config,
                current_stage=current_stage,
            )
        ]
        params["transformers"] = transformers

        return self._component_class(**params)

    def get_valid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        transformers = self.components
        transformers = [
            (
                name,
                recursively_remove_invalid_components(
                    transformer,
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                ),
                columns,
            )
            for name, transformer, columns in transformers
            if is_component_valid_iterable(
                transformer,
                pipeline_config=pipeline_config,
                current_stage=current_stage,
            )
        ]
        return transformers

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = super().get_tuning_grid(use_extended=use_extended)
        transformer_grids = {
            name: get_step_choice_grid(transformer)
            for name, transformer, columns in self.final_parameters["transformers"]
        }
        return {**transformer_grids, **default_grid}