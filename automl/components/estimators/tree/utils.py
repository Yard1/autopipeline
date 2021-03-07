import numpy as np

from ....search.distributions import IntUniformDistribution

# https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/decision_tree.py
def estimate_max_depth(config, stage) -> IntUniformDistribution:
    X = config.X
    if X is None:
        return IntUniformDistribution(1, 15)
    num_features = X.shape[1]
    max_max_depth_factor = 2.5
    max_max_depth_factor = max(2, int(np.round(max_max_depth_factor * num_features, 0)))
    return IntUniformDistribution(1, int(max_max_depth_factor))
