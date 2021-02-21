from ...utils.string import removeprefix


class PrefixParamsMixin:
    def set_params(self, **params):
        params = {removeprefix(k, f"{self._automl_prefix}_"): v for k, v in params.items()}
        return super().set_params(**params)