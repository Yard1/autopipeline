import numpy as np
import pandas as pd

from ray import tune

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter

from ..utils import call_component_if_needed
from ...components import Component
from ...utils.string import removesuffix


class Tuner:
    def __init__(self) -> None:
        pass


def remove_component_suffix(key: str):
    split_key = [s for s in key.split("__") if s[-1] != Component._automl_id_sign]
    return "__".join(split_key)


class RayTuneTuner(Tuner):
    def _trial_with_cv(self, config, checkpoint_dir=None):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config_called = {
            remove_component_suffix(k): call_component_if_needed(
                v, random_state=self.random_state
            )
            for k, v in config.items()
        }

        estimator.set_params(**config_called)

        for fraction in self.early_stopping_fractions_:
            if len(self.early_stopping_fractions_) > 1:
                subsample_cv = _SubsampleMetaSplitter(
                    base_cv=self.cv,
                    fraction=fraction,
                    subsample_test=True,
                    random_state=self.random_state,
                )
            else:
                subsample_cv = self.cv
            scores = cross_validate(
                estimator,
                self.X_,
                self.y_,
                cv=subsample_cv,
                # error_score=self.error_score,
                # fit_params=self.fit_params,
                # groups=self.groups,
                # return_train_score=self.return_train_score,
                # scoring=self.scoring,
            )

            tune.report(mean_test_score=np.mean(scores["test_score"]))