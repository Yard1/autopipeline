import ray


class ray_context:
    DEFAULT_CONFIG = {
        "ignore_reinit_error": True,
        "configure_logging": False,
        "include_dashboard": False,
    }

    def __init__(self, **ray_config):
        self.ray_config = {**self.DEFAULT_CONFIG, **ray_config}
        self.ray_init = False

    def __enter__(self):
        self.ray_init = ray.is_initialized()
        if not self.ray_init:
            ray.init(
                **self.ray_config
                # log_to_driver=self.verbose == 2
            )

    def __exit__(self, type, value, traceback):
        if not self.ray_init and ray.is_initialized():
            ray.shutdown()