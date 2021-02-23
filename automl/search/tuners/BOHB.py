import logging
from copy import deepcopy
import traceback


import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import statsmodels.api as sm

from hpbandster.optimizers.config_generators.bohb import BOHB


class BOHBConditional(BOHB):
    def get_config(self, budget):
        """
        Function to sample a new configuration

        This function is called inside Hyperband to query a new configuration


        Parameters:
        -----------
        budget: float
            the budget for which this configuration is scheduled

        returns: config
            should return a valid configuration

        """

        self.logger.debug("start sampling a new configuration.")

        sample = None
        info_dict = {}

        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            sample = self.configspace.sample_configuration()
            info_dict["model_based_pick"] = False

        best = np.inf
        best_vector = None
        best_vectors = []

        if sample is None:
            try:

                # sample from largest budget
                budget = max(self.kde_models.keys())

                l = self.kde_models[budget]["good"].pdf
                g = self.kde_models[budget]["bad"].pdf

                minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                kde_good = self.kde_models[budget]["good"]
                kde_bad = self.kde_models[budget]["bad"]

                for i in range(self.num_samples):
                    idx = np.random.randint(0, len(kde_good.data))
                    datum = kde_good.data[idx]
                    vector = []

                    for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                        bw = max(bw, self.min_bandwidth)
                        if t == 0:
                            bw = self.bw_factor * bw
                            try:
                                vector.append(
                                    sps.truncnorm.rvs(
                                        -m / bw, (1 - m) / bw, loc=m, scale=bw
                                    )
                                )
                            except:
                                self.logger.warning(
                                    "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s"
                                    % (datum, kde_good.bw, m)
                                )
                                self.logger.warning(
                                    "data in the KDE:\n%s" % kde_good.data
                                )
                        else:

                            if np.random.rand() < (1 - bw):
                                vector.append(int(m))
                            else:
                                vector.append(np.random.randint(t))
                    val = minimize_me(vector)

                    if not np.isfinite(val):
                        self.logger.warning(
                            "sampled vector: %s has EI value %s" % (vector, val)
                        )
                        self.logger.warning(
                            "data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data)
                        )
                        self.logger.warning(
                            "bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw)
                        )
                        self.logger.warning("l(x) = %s" % (l(vector)))
                        self.logger.warning("g(x) = %s" % (g(vector)))

                        # right now, this happens because a KDE does not contain all values for a categorical parameter
                        # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                        # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                        if np.isfinite(l(vector)):
                            best_vectors.append((-np.inf, vector))
                            best_vector = vector
                            break

                    if val < best:
                        best_vectors.append((val, vector))
                        best = val
                        best_vector = vector

                if best_vector is None:
                    self.logger.debug(
                        "Sampling based optimization with %i samples failed -> using random configuration"
                        % self.num_samples
                    )
                    sample = self.configspace.sample_configuration().get_dictionary()
                    info_dict["model_based_pick"] = False
                else:
                    self.logger.debug(
                        "best_vector: {}, {}, {}, {}".format(
                            best_vector, best, l(best_vector), g(best_vector)
                        )
                    )
                    success = False
                    for val, best_vector in sorted(best_vectors, key=lambda x: x[0]):
                        for i, hp_value in enumerate(best_vector):
                            if isinstance(
                                self.configspace.get_hyperparameter(
                                    self.configspace.get_hyperparameter_by_idx(i)
                                ),
                                ConfigSpace.hyperparameters.CategoricalHyperparameter,
                            ):
                                best_vector[i] = int(np.rint(best_vector[i]))
                        sample = ConfigSpace.Configuration(
                            self.configspace, vector=best_vector
                        ).get_dictionary()

                        try:
                            sample = (
                                ConfigSpace.util.deactivate_inactive_hyperparameters(
                                    configuration_space=self.configspace,
                                    configuration=sample,
                                )
                            )
                            info_dict["model_based_pick"] = True
                            success = True
                            break

                        except Exception as e:
                            self.logger.warning(
                                ("=" * 50 + "\n") * 3
                                + "Error converting configuration:\n%s" % sample
                                + "\n here is a traceback:"
                                + traceback.format_exc()
                            )
                    assert success
            except:
                self.logger.warning(
                    "Sampling based optimization with %i samples failed\n %s \nUsing random configuration"
                    % (self.num_samples, traceback.format_exc())
                )
                sample = self.configspace.sample_configuration()
                info_dict["model_based_pick"] = False

        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary(),
            ).get_dictionary()
        except Exception as e:
            self.logger.warning(
                "Error (%s) converting configuration: %s -> "
                "using random configuration!",
                e,
                sample,
            )
            sample = self.configspace.sample_configuration().get_dictionary()
        self.logger.debug("done sampling a new configuration.")
        return sample, info_dict
