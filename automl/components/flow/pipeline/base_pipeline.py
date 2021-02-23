from typing import List, Optional

from copy import copy
from sklearn.base import clone
from imblearn.pipeline import Pipeline as _ImblearnPipeline
from sklearn.pipeline import Pipeline as _Pipeline, _fit_transform_one
from sklearn.utils import (
    Bunch,
    _print_elapsed_time,
)
from sklearn.utils.validation import check_memory

from ..flow import Flow
from ..utils import (
    recursively_remove_invalid_components,
    get_single_component_from_iterable,
    is_component_valid_iterable,
    get_step_choice_grid,
)
from ..utils import convert_tuning_grid
from ...transformers import *
from ...estimators import *
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ....search.distributions import CategoricalDistribution


class BasePipeline(_Pipeline):
    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx, name, transformer) in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location"):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, "cachedir"):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            result = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            if len(result) == 3:
                X, y, fitted_transformer = result
            else:
                X, fitted_transformer = result
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, y = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Transformed samples
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, y = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)


class Pipeline(Flow):
    _component_class = _ImblearnPipeline

    _default_parameters = {
        "memory": None,
        "verbose": False,
    }

    @property
    def components_name(self) -> str:
        return "steps"

    def get_default_components_configuration(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ):
        steps = [
            (
                name,
                get_single_component_from_iterable(
                    step, pipeline_config=pipeline_config, current_stage=current_stage
                ),
            )
            for name, step in self.components
            if is_component_valid_iterable(
                step, pipeline_config=pipeline_config, current_stage=current_stage
            )
        ]
        return steps

    def __call__(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
    ):
        params = self.final_parameters.copy()
        steps = self.get_default_components_configuration(
            pipeline_config=pipeline_config,
            current_stage=current_stage,
        )
        steps = [
            (
                name,
                component(
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                    random_state=random_state,
                ),
            )
            for name, component in steps
        ]
        params["steps"] = steps

        return self._component_class(**params)

    def get_valid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        steps = self.components
        steps = [
            (
                name,
                recursively_remove_invalid_components(
                    step, pipeline_config=pipeline_config, current_stage=current_stage
                ),
            )
            for name, step in steps
            if is_component_valid_iterable(
                step, pipeline_config=pipeline_config, current_stage=current_stage
            )
        ]
        return steps

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = super().get_tuning_grid(use_extended=use_extended)
        step_grids = {
            name: get_step_choice_grid(step)
            for name, step in self.final_parameters["steps"]
        }
        return {**step_grids, **default_grid}

    def __copy__(self):
        # self.spam is to be ignored, it is calculated anew for the copy
        # create a new copy of ourselves *reusing* self.bar
        new = type(self)(tuning_grid=self.tuning_grid, **self.parameters)
        new.components = self.components.copy()
        new.components = [
            (
                copy(name),
                copy(step) if isinstance(step, (list, dict, tuple, Flow)) else step,
            )
            for name, step in new.components
        ]
        return new


class TopPipeline(Pipeline):
    def get_estimator_distribution(self):
        return (self.components[-1][0], CategoricalDistribution(self.components[-1][1]))

    def get_preprocessor_distribution(self):
        grid = self.get_tuning_grid()
        grid.pop(self.components[-1][0])
        return {
            k: CategoricalDistribution(v) for k, v in convert_tuning_grid(grid).items()
        }

    def get_all_distributions(self):
        grid = self.get_tuning_grid()
        return {
            k: CategoricalDistribution(v) for k, v in convert_tuning_grid(grid).items()
        }

    def remove_invalid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        super().remove_invalid_components(
            pipeline_config=pipeline_config, current_stage=current_stage
        )
        self._remove_invalid_preset_components_configurations(
            pipeline_config=pipeline_config, current_stage=current_stage
        )
        return self

    def _remove_invalid_preset_components_configurations(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ) -> list:
        converted_configurations = []
        for configuration in self.preset_configurations:
            estimator, estimator_parameters = configuration["Estimator"]
            if not estimator.is_component_valid(
                config=pipeline_config, stage=current_stage
            ):
                continue

            if pipeline_config is None:
                config = ComponentConfig(estimator=estimator)
            else:
                config = copy(pipeline_config)
                config.estimator = estimator

            hyperparams = {}
            components_to_remove = []

            for name, component_parameters in configuration.items():
                component, parameters = component_parameters
                if component != estimator and not component.is_component_valid(
                    config=pipeline_config, stage=current_stage
                ):
                    components_to_remove.append(name)
                    continue
                default_hyperparameters = {
                    k: v.default for k, v in component.get_tuning_grid().items()
                }

                diff = set(parameters.keys()) - set(default_hyperparameters.keys())
                if diff:
                    raise KeyError(
                        f"parameter names {list(diff)} are not present in _default_tuning_grid for {component}"
                    )

                parameters = {**default_hyperparameters, **parameters}
                parameters = {
                    component.get_hyperparameter_key_suffix(
                        name, parameter_name
                    ): parameter
                    for parameter_name, parameter in parameters.items()
                }
                hyperparams = {**hyperparams, **parameters}

                configuration[name] = component

            for name in components_to_remove:
                configuration.pop(name)
            configuration = {**configuration, **hyperparams}
            if configuration not in converted_configurations:
                converted_configurations.append(configuration)
        self.preset_configurations = converted_configurations

    def __init__(
        self,
        tuning_grid=None,
        preset_configurations: Optional[List[dict]] = None,
        **parameters,
    ) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}
        self.preset_configurations = preset_configurations or []
        assert "steps" in self.parameters
        assert "Estimator" == self.parameters["steps"][-1][0]
