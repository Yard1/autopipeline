import ray
import os
from unittest.mock import patch
import contextlib

from typing import Optional, Union, Dict, List, Tuple
from collections import defaultdict

from copy import copy

from ...components.component import ComponentConfig
from ...search.stage import AutoMLStage
from ...components.flow.pipeline import Pipeline
from ...utils.string import removeprefix


class ray_context:
    DEFAULT_CONFIG = {
        "ignore_reinit_error": True,
        "configure_logging": False,
        "include_dashboard": False,
        # "local_mode": True,
        # "num_cpus": 1,
    }

    def __init__(self, global_checkpoint_s=10, **ray_config):
        self.global_checkpoint_s = global_checkpoint_s
        self.ray_config = {**self.DEFAULT_CONFIG, **ray_config}
        self.ray_init = False

    def __enter__(self):
        self.ray_init = ray.is_initialized()
        if not self.ray_init:
            with patch.dict(
                "os.environ",
                {"TUNE_GLOBAL_CHECKPOINT_S": str(self.global_checkpoint_s)},
            ) if "TUNE_GLOBAL_CHECKPOINT_S" not in os.environ else contextlib.nullcontext():
                ray.init(
                    **self.ray_config
                    # log_to_driver=self.verbose == 2
                )

    def __exit__(self, type, value, traceback):
        if not self.ray_init and ray.is_initialized():
            ray.shutdown()


def split_list_into_chunks(lst: list, chunk_size: int):
    return [
        lst[i * chunk_size : (i + 1) * chunk_size]
        for i in range((len(lst) + chunk_size - 1) // chunk_size)
    ]


def get_conditions(spec: Dict, to_str=False) -> dict:
    spec = copy(spec)
    conditions_spec = defaultdict(dict)
    estimator_name, estimators = spec.get_estimator_distribution()

    for estimator in estimators.values:
        for k2, v2 in estimator.get_tuning_grid().items():
            name = estimator.get_hyperparameter_key_suffix(estimator_name, k2)
            if name not in conditions_spec[estimator_name]:
                conditions_spec[estimator_name][name] = []
            conditions_spec[estimator_name][name].append(
                str(estimator) if to_str else estimator
            )

    preprocessors_grid = spec.get_preprocessor_distribution()

    for estimator in estimators.values:
        spec_copy = copy(spec)
        spec_copy.remove_invalid_components(
            pipeline_config=ComponentConfig(estimator=estimator),
            current_stage=AutoMLStage.TUNE,
        )
        grid = spec_copy.get_preprocessor_distribution()
        removed_components = {
            k: [v2 for v2 in v.values if v2 not in grid[k].values]
            if k in grid
            else v.values
            for k, v in preprocessors_grid.items()
        }
        print(f"{estimator}: {removed_components}")
        for k, v in preprocessors_grid.items():
            for v2 in v.values:
                if v2 not in removed_components[k]:
                    if k not in conditions_spec[estimator_name]:
                        conditions_spec[estimator_name][k] = []
                    conditions_spec[estimator_name][k].append(
                        str(estimator) if to_str else estimator
                    )

    for k, v in preprocessors_grid.items():
        for choice in v.values:
            for k2, v2 in choice.get_tuning_grid().items():
                name = choice.get_hyperparameter_key_suffix(k, k2)
                if name not in conditions_spec[k]:
                    conditions_spec[k][name] = []
                conditions_spec[k][name].append(str(choice) if to_str else choice)

    return conditions_spec


def enforce_conditions_on_config(
    config, conditional_space, prefix="", keys_to_keep=None
):
    config = config.copy()
    for independent_name, v in conditional_space.items():
        for dependent_name, required_values in v.items():
            if f"{prefix}{independent_name}" not in config or (
                config[f"{prefix}{independent_name}"] not in required_values
                and f"{prefix}{dependent_name}" in config
            ):
                config.pop(f"{prefix}{dependent_name}", None)
    if keys_to_keep:
        config = {k: v for k, v in config.items() if not k.startswith(prefix) or removeprefix(k, prefix) in keys_to_keep}
    return config


def get_all_tunable_params(pipeline: Pipeline, to_str=False) -> Tuple[dict, dict]:
    space = {k: v for k, v in pipeline.get_all_distributions().items()}
    string_space = {}
    for k, v in space.items():
        choices = v.values
        for choice in choices:
            string_space[str(choice)] = choice
    hyperparams = {}
    for k, v in space.items():
        for v2 in v.values:
            for k3, v3 in v2.get_tuning_grid().items():
                name = v2.get_hyperparameter_key_suffix(k, k3)
                hyperparams[name] = v3
        if to_str:
            v.values = [str(x) for x in v.values]
    space = {**space, **hyperparams}

    return space, string_space
