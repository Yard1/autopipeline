import ray
import os
from unittest.mock import patch
import contextlib

from typing import Optional, Union, Dict, List, Tuple

from ..components.component import Component


def call_component_if_needed(possible_component, **kwargs):
    if isinstance(possible_component, Component):
        return possible_component(**kwargs)
    else:
        return possible_component


class ray_context:
    DEFAULT_CONFIG = {
        "ignore_reinit_error": True,
        "configure_logging": False,
        "include_dashboard": True,
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