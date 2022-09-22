from unittest.mock import patch
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.samplers import TPESampler as _TPESampler
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator,
    EPS,
    SIGMA0_MAGNITUDE,
)
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class ParzenEstimator(_ParzenEstimator):
    def _precompute_sigmas0(
        self, observations: Dict[str, np.ndarray]
    ) -> Optional[float]:
        n_observations = next(iter(observations.values())).size
        n_observations = max(n_observations, 1)

        # If it is univariate, there is no need to precompute sigmas0, so this method returns None.
        if not self._parameters.multivariate:
            return None

        # We use Scott's rule for bandwidth selection if the number of parameters > 1.
        # This rule was used in the BOHB paper.
        # TODO(kstoneriv3): The constant factor SIGMA0_MAGNITUDE=0.2 might not be optimal.
        obs = np.array(list(observations.values()))
        sigmas = np.std(obs, ddof=1, axis=0)
        IQRs = np.subtract.reduce(np.percentile(obs, [75, 25], axis=0))
        bandwidths = 1.059 * np.minimum(sigmas, IQRs) * np.power(n_observations, -0.2)
        return bandwidths

    def _calculate_numerical_params(
        self, observations: np.ndarray, param_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_observations = self._n_observations
        consider_prior = self._parameters.consider_prior
        consider_endpoints = self._parameters.consider_endpoints
        consider_magic_clip = self._parameters.consider_magic_clip
        multivariate = self._parameters.multivariate
        sigmas0 = self._sigmas0
        low = self._low[param_name]
        high = self._high[param_name]
        assert low is not None
        assert high is not None
        assert len(observations) == self._n_observations

        if n_observations == 0:
            consider_prior = True

        prior_mu = 0.5 * (low + high)
        prior_sigma = 1.0 * (high - low)

        if consider_prior:
            mus = np.empty(n_observations + 1)
            mus[:n_observations] = observations
            mus[n_observations] = prior_mu
            sigmas = np.empty(n_observations + 1)
        else:
            mus = observations
            sigmas = np.empty(n_observations)

        if multivariate:
            assert sigmas0 is not None
            print(sigmas0)
            sigmas[:-1] = sigmas0
            print(sigmas)
        else:
            assert sigmas0 is None
            sorted_indices = np.argsort(mus)
            sorted_mus = mus[sorted_indices]
            sorted_mus_with_endpoints = np.empty(len(mus) + 2, dtype=float)
            sorted_mus_with_endpoints[0] = low
            sorted_mus_with_endpoints[1:-1] = sorted_mus
            sorted_mus_with_endpoints[-1] = high

            sorted_sigmas = np.maximum(
                sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
            )

            if not consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                sorted_sigmas[0] = (
                    sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                )
                sorted_sigmas[-1] = (
                    sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]
                )

            sigmas[:] = sorted_sigmas[np.argsort(sorted_indices)]

        # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(mus)))
        else:
            minsigma = EPS
        sigmas = np.asarray(np.clip(sigmas, minsigma, maxsigma))

        if consider_prior:
            sigmas[n_observations] = prior_sigma

        return mus, sigmas


class TPESampler(_TPESampler):
    def _sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        with patch("optuna.samplers._tpe.sampler._ParzenEstimator", ParzenEstimator):
            return super()._sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        with patch("optuna.samplers._tpe.sampler._ParzenEstimator", ParzenEstimator):
            return super().sample_independent(
                study, trial, param_name, param_distribution
            )
