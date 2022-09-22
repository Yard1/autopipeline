from sklearn.ensemble import (
    RandomForestClassifier as _RandomForestClassifier,
    RandomForestRegressor as _RandomForestRegressor,
    ExtraTreesClassifier as _ExtraTreesClassifier,
    ExtraTreesRegressor as _ExtraTreesRegressor,
)


class RandomForestExtraTreesMixin:
    def fit(self, X, y, sample_weight=None):
        assert self.randomization_type in ("rf", "et")
        if (
            self.randomization_type == "et"
            and not type(self._estimator) is self._ET_CLASS
        ) or (
            self.randomization_type == "rf"
            and not type(self._estimator) is self._RF_CLASS
        ):
            self._create_estimator()
        return self._estimator.fit(X=X, y=y, sample_weight=sample_weight)

    def _create_estimator(self, **params):
        if params:
            if self.randomization_type == "et":
                self._estimator = self._ET_CLASS(**params)
            else:
                self._estimator = self._RF_CLASS(**params)
        else:
            if self.randomization_type == "et":
                self._estimator = self._ET_CLASS(
                    **self._estimator.get_params(deep=False)
                )
            else:
                self._estimator = self._RF_CLASS(
                    **self._estimator.get_params(deep=False)
                )

    def get_underlying_estimator(self):
        return self._estimator

    def __getattribute__(self, name: str):
        if (
            name
            in (
                "randomization_type",
                "_estimator",
                "get_underlying_estimator",
                "set_params",
                "get_params",
                "_create_estimator",
                "fit",
                "_ET_CLASS",
                "_RF_CLASS",
            )
            or name.startswith("__")
            and name not in ("__repr__", "__str__", "__getattribute__")
        ):
            return super().__getattribute__(name)
        return self._estimator.__getattribute__(name)

    def set_params(self, **params):
        if "randomization_type" in params:
            self.randomization_type = params.pop("randomization_type")
        if params.get("max_samples", None) == 1.0:
            params["max_samples"] = None
        return self._estimator.set_params(**params)

    def get_params(self, deep: bool = True):
        r = {
            "randomization_type": self.randomization_type,
            **self._estimator.get_params(deep=deep),
        }
        if "max_samples" in r and r["max_samples"] is None:
            r["max_samples"] = 1.0
        return r


class RandomForestExtraTreesClassifier(
    RandomForestExtraTreesMixin, _RandomForestClassifier
):
    _ET_CLASS = _ExtraTreesClassifier
    _RF_CLASS = _RandomForestClassifier

    def __init__(
        self,
        n_estimators=100,
        *,
        randomization_type="rf",
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None
    ):
        assert randomization_type in ("rf", "et")
        self.randomization_type = randomization_type
        if max_samples == 1.0:
            max_samples = None
        self._create_estimator(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )


class RandomForestExtraTreesRegressor(
    RandomForestExtraTreesMixin, _RandomForestRegressor
):
    _ET_CLASS = _ExtraTreesRegressor
    _RF_CLASS = _RandomForestRegressor

    def __init__(
        self,
        n_estimators=100,
        *,
        randomization_type="rf",
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None
    ):
        assert randomization_type in ("rf", "et")
        self.randomization_type = randomization_type
        if max_samples == 1.0:
            max_samples = None
        self._create_estimator(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
