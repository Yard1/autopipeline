from typing import Optional, Union, Dict, List, Tuple
from collections import defaultdict

from copy import copy, deepcopy

from ...components.component import ComponentConfig, Component
from ...search.stage import AutoMLStage
from ...components.flow.pipeline import Pipeline
from ...utils.string import removeprefix

from ..utils import call_component_if_needed

def remove_component_suffix(key: str):
    split_key = [s for s in key.split("__") if s[-1] != Component._automl_id_sign]
    return "__".join(split_key)


def split_list_into_chunks(lst: list, chunk_size: int):
    return [
        lst[i * chunk_size : (i + 1) * chunk_size]
        for i in range((len(lst) + chunk_size - 1) // chunk_size)
    ]


def treat_config(config, component_strings, random_state=None):
    config = {k: component_strings.get(v, v) for k, v in config.items()}
    return {
        remove_component_suffix(k): call_component_if_needed(
            v, random_state=random_state
        )
        for k, v in config.items()
    }

def get_conditions(spec: Dict, to_str=False, use_extended=False) -> dict:
    spec = copy(spec)
    conditions_spec = defaultdict(dict)
    estimator_name, estimators = spec.get_estimator_distribution()

    preprocessors_grid = spec.get_preprocessor_distribution()

    conditions_spec[estimator_name] = defaultdict(dict)

    for estimator in estimators.values:
        spec_copy = copy(spec)
        spec_copy.remove_invalid_components(
            pipeline_config=ComponentConfig(estimator=estimator),
            current_stage=AutoMLStage.TUNE,
        )
        grid = spec_copy.get_preprocessor_distribution()
        remaining_components = {
            k: [v2 for v2 in v.values if v2 in grid[k].values] if k in grid else []
            for k, v in preprocessors_grid.items()
        }
        # print(f"{estimator}: {removed_components}")
        estimator_key = str(estimator) if to_str else estimator
        conditions_spec[estimator_name][estimator_key] = (
            {k: [str(x) for x in v] for k, v in remaining_components.items()}
            if to_str
            else remaining_components
        )
        for k2, v2 in estimator.get_tuning_grid(use_extended=use_extended).items():
            name = estimator.get_hyperparameter_key_suffix(estimator_name, k2)
            conditions_spec[estimator_name][estimator_key][name] = True

    for k, v in preprocessors_grid.items():
        conditions_spec[k] = defaultdict(dict)
        for choice in v.values:
            for k2, v2 in choice.get_tuning_grid(use_extended=use_extended).items():
                name = choice.get_hyperparameter_key_suffix(k, k2)
                choice_key = str(choice) if to_str else choice
                conditions_spec[k][choice_key][name] = True

    return conditions_spec


def enforce_conditions_on_config(
    config, conditional_space, prefix="", keys_to_keep=None, raise_exceptions=True
):
    config = config.copy()
    allowed_keys = {"Estimator"}
    for param_key, independent_choice in conditional_space.items():
        if f"{prefix}{param_key}" not in config or param_key not in allowed_keys:
            continue
        space = independent_choice[config[f"{prefix}{param_key}"]]
        allowed_keys.update(space.keys())
        for conditional_param_key, allowed_values in space.items():
            prefixed_key = f"{prefix}{conditional_param_key}"
            if allowed_values is not True:
                if len(allowed_values) == 1:
                    config[prefixed_key] = allowed_values[0]
                elif len(allowed_values) == 0:
                    config[prefixed_key] = "passthrough"
            if raise_exceptions:
                if allowed_values is True:
                    assert prefixed_key in config, f"{prefixed_key} not in {config}"
                elif len(allowed_values) > 0:
                    assert (
                        config[prefixed_key] in allowed_values
                    ), f"{config[prefixed_key]} not in {allowed_values}"
                elif len(allowed_keys) == 0:
                    assert (
                        config[prefixed_key] == "passthrough"
                    ), f"{config[prefixed_key]} is not 'passthrough'"

    if keys_to_keep:
        allowed_keys = {k for k in allowed_keys if k in keys_to_keep}
    allowed_keys = {f"{prefix}{k}" for k in allowed_keys}
    if prefix:
        allowed_keys.update({k for k in config if not k.startswith(prefix)})
    config = {k: v for k, v in config.items() if k in allowed_keys}
    return config


def get_all_tunable_params(
    pipeline: Pipeline, to_str=False, use_extended=False, space=None
) -> Tuple[dict, dict]:
    if space is None:
        space = {
            k: v
            for k, v in pipeline.get_all_distributions(
                use_extended=use_extended
            ).items()
        }
    string_space = {}
    for k, v in space.items():
        choices = v.values
        for choice in choices:
            string_space[str(choice)] = choice
    hyperparams = {}
    for k, v in space.items():
        for v2 in v.values:
            for k3, v3 in v2.get_tuning_grid(use_extended=use_extended).items():
                name = v2.get_hyperparameter_key_suffix(k, k3)
                hyperparams[name] = v3
        if to_str:
            v.values = list({str(x) for x in v.values})
    space = {**space, **hyperparams}

    return space, string_space
