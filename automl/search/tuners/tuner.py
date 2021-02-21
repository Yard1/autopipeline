import numpy as np
import pandas as pd

from ray import tune

from sklearn.model_selection import cross_validate

from ..utils import call_component_if_needed


class Tuner:
    def __init__(self) -> None:
        pass


class RayTuneTuner(Tuner):
    def _trial_with_cv(self, config):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config_called = {
            k: call_component_if_needed(
                v, random_state=self.random_state, return_prefix_mixin=True
            )
            for k, v in config.items()
        }

        estimator.set_params(**config_called)

        scores = cross_validate(
            estimator,
            self.X_,
            self.y_,
            # cv=self.cv,
            # error_score=self.error_score,
            # fit_params=self.fit_params,
            # groups=self.groups,
            # return_train_score=self.return_train_score,
            # scoring=self.scoring,
        )

        tune.report(mean_test_score=np.mean(scores["test_score"]))