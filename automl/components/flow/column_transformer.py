from copy import deepcopy, copy
from sklearn.compose import ColumnTransformer as _ColumnTransformer

from .flow import Flow
from .utils import (
    recursively_remove_invalid_components,
    get_single_component_from_iterable,
    is_component_valid_iterable,
    get_step_choice_grid,
    recursively_call_tuning_grid_funcs,
)
from ..component import ComponentConfig
from ...search.stage import AutoMLStage

import numpy as np
import pandas as pd
from scipy import sparse


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

    def get_default_components_configuration(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ):
        transformers = [
            (
                name,
                get_single_component_from_iterable(
                    transformer,
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                ),
                columns,
            )
            for name, transformer, columns in self.components
            if is_component_valid_iterable(
                transformer,
                pipeline_config=pipeline_config,
                current_stage=current_stage,
            )
        ]
        return transformers

    def __call__(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
    ):
        params = self.final_parameters.copy()
        transformers = self.get_default_components_configuration(
            pipeline_config=pipeline_config,
            current_stage=current_stage,
        )
        transformers = [
            (
                name,
                component(
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                    random_state=random_state,
                ),
                columns,
            )
            for name, component, columns in transformers
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
            name: get_step_choice_grid(transformer, use_extended=use_extended)
            for name, transformer, columns in self.components
        }
        return {**transformer_grids, **default_grid}

    def call_tuning_grid_funcs(self, config: ComponentConfig, stage: AutoMLStage, use_extended: bool = False):
        super().call_tuning_grid_funcs(config, stage, use_extended=use_extended)
        for name, transformer, columns in self.components:
            recursively_call_tuning_grid_funcs(transformer, config=config, stage=stage, use_extended=use_extended)

    def __copy__(self):
        # self.spam is to be ignored, it is calculated anew for the copy
        # create a new copy of ourselves *reusing* self.bar
        new = type(self)(tuning_grid=self.tuning_grid, **self.parameters)
        new.components = self.components.copy()
        new.components = [
            (
                copy(name),
                copy(transformer)
                if isinstance(transformer, (list, dict, tuple, Flow))
                else transformer,
                copy(columns),
            )
            for name, transformer, columns in new.components
        ]
        return new
