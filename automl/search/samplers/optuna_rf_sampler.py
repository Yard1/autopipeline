import abc
import copy
from datetime import datetime
from typing import Callable, Optional, Tuple, Type, Any, List, Dict, Sequence, Union
import io
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr, contextmanager

import numpy as np
import pandas as pd
from scipy.stats import norm

import optuna
from optuna import distributions
from optuna import samplers
from optuna._imports import try_import
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, Trial
from optuna.trial import TrialState
from optuna.samplers import BaseSampler, RandomSampler
from optuna.samplers._search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna._transform import _SearchSpaceTransform

from sklearn.preprocessing import minmax_scale, power_transform
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from category_encoders import GLMMEncoder
from category_encoders.wrapper import NestedCVWrapper
from automl_models.components.estimators.tree.gradient_booster import (
    CatBoostRegressorWithAutoCatFeatures,
)

from .random_forest import RandomForestRegressorWithStd

EPS = 1e-10


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


class FastTrial(FrozenTrial):
    def _validate(self) -> None:
        return

    def _suggest(self, name: str, distribution: distributions.BaseDistribution) -> Any:

        if name not in self._params:
            search_space = {name: distribution}
            trans = _SearchSpaceTransform(search_space)
            trans_params = self.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])

            self._params[name] = trans.untransform(trans_params)[name]

        value = self._params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(value)
        if not distribution._contains(param_value_in_internal_repr):
            raise ValueError(
                "The value {} of the parameter '{}' is out of "
                "the range of the distribution {}.".format(value, name, distribution)
            )

        if name in self._distributions:
            distributions.check_distribution_compatibility(
                self._distributions[name], distribution
            )

        self._distributions[name] = distribution

        return value


def _run_trial(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    rng,
) -> Tuple[Dict[str, Any], float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        optuna.storages.fail_stale_trials(study)

    trial = FastTrial(
        number=-1,
        trial_id=-1,
        state=TrialState.RUNNING,
        value=None,
        values=None,
        datetime_start=datetime.now(),
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    trial.rng = rng

    value = func(trial)

    return trial.params, value


class BaseSamplerModel(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tell(
        self,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def ask(
        self,
        trial: FrozenTrial,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _complete_trial_to_observation(
        self,
        trial: FrozenTrial,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        return self._complete_to_observation(trial.params, trial.value)

    def _complete_to_observation(
        self, params: Dict[str, Any], value: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:

        param_values = {}
        categorical_param_values = {}
        for name, distribution in sorted(self.search_space.items()):
            param_value = params.get(name, None)

            if isinstance(distribution, distributions.CategoricalDistribution):
                categorical_param_values[name] = param_value
                continue

            if param_value:
                if isinstance(distribution, distributions.DiscreteUniformDistribution):
                    param_value = (param_value - distribution.low) // distribution.q
                elif isinstance(distribution, distributions.IntUniformDistribution):
                    param_value = (param_value - distribution.low) // distribution.step

            param_values[name] = param_value

        return param_values, categorical_param_values, value


def ei(par):
    def _func(
        best_observation_value,
        predicted_value,
        predicted_std,
        noise=0,
        random_state=None,
    ):

        py, ps2 = predicted_value, predicted_std
        ps = np.sqrt(ps2)
        normed = (
            best_observation_value
            - EPS
            - py
            - (
                0
                if not noise
                else np.sqrt(2.0 * noise)
                * norm.rvs(size=py.shape, random_state=random_state)
            )
        ) / ps
        phi = norm.pdf(normed)
        Phi = norm.cdf(normed)
        EI = ps * (Phi * normed + phi)
        return EI

    return _func


def logei(par):
    ei_f = ei(par)

    def _func(
        best_observation_value,
        predicted_value,
        predicted_std,
        noise=0,
        random_state=None,
    ):
        with np.errstate(divide="ignore"):
            return np.log(
                ei_f(
                    best_observation_value,
                    predicted_value,
                    predicted_std,
                    noise=noise,
                    random_state=random_state,
                )
            )

    return _func


class RandomForestSamplerModel(BaseSamplerModel):
    def __init__(
        self,
        study: Study,
        distributions_function: Callable[[Trial], None],
        search_space: Dict[str, distributions.BaseDistribution],
        n_ei_candidates: int,
        best_value: float,
        independent_sampler: Union[BaseSampler, Type[BaseSampler]],
        *,
        acq_function: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = logei(0),
        n_estimators: int = 10,
        boostrap: bool = True,
        max_features: Union[str, float] = "auto",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        independent_sampler_kwargs: Optional[Dict[str, Any]] = None,
        early_stopping_patience: int = 50,
        early_stopping_delay: int = 10,
    ) -> None:
        self.study = study
        self.search_space = search_space
        self.search_space_bounds = self._get_search_space_bounds()
        self.acq_function = acq_function
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.bootstrap = boostrap
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        self.best_value = best_value
        self.n_ei_candidates = n_ei_candidates
        self.independent_sampler = independent_sampler(
            seed=random_state, **independent_sampler_kwargs
        )
        self.distributions_function = distributions_function
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delay = early_stopping_delay

        self._rng = np.random.RandomState(self.random_state)
        self._model = RandomForestRegressorWithStd(
            n_estimators=n_estimators,
            bootstrap=boostrap,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._encoder = NestedCVWrapper(
            GLMMEncoder(random_state=random_state, binomial_target=False),
            cv=KFold(5, shuffle=True, random_state=random_state),
        )
        self._noise = 0

        with redirect_stdout(io.StringIO()), redirect_stderr(
            io.StringIO()
        ), all_logging_disabled():
            self._independent_study = optuna.create_study(
                storage=optuna.storages.InMemoryStorage(),
                sampler=self.independent_sampler,
                direction="maximize",
            )

    def _get_search_space_bounds(self) -> Dict[str, Tuple[float, float]]:
        search_space_bounds = {}
        for name, distribution in sorted(self.search_space.items()):
            if isinstance(distribution, distributions.CategoricalDistribution):
                continue

            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                low = 0
                high = (distribution.high - distribution.low) // distribution.q
            elif isinstance(distribution, distributions.IntUniformDistribution):
                low = 0
                high = (distribution.high - distribution.low) // distribution.step
            else:
                low, high = distribution.low, distribution.high

            search_space_bounds[name] = (low, high)
        return search_space_bounds

    def _get_model_preds(self, xs: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return self._model.predict(xs)

    def clone(self, **params) -> "RandomForestSamplerModel":
        return RandomForestSamplerModel(
            {**{k: v for k, v in self.__dict__ if not k.startswith("_")}, **params}
        )

    # def ask(
    #     self,
    #     trial: FrozenTrial,
    # ) -> Dict[str, Any]:
    #     # xs, _ = self._preprocess_trials(
    #     #     [(trial.params, trial.value) for trial in self._complete_trials], fit=False
    #     # )
    #     # x_pred, x_var = self._get_model_preds(xs)
    #     # acq = self.acq_function(self.best_value, x_pred, x_var)
    #     # trials = [
    #     #     optuna.create_trial(
    #     #         value=acq[i],
    #     #         params=self._complete_trials[i].params,
    #     #         distributions=self._complete_trials[i].distributions,
    #     #     )
    #     #     for i in range(len(acq))
    #     # ]
    #     # self._independent_study.add_trials(trials)

    #     def opt_func(trial: Trial):
    #         self.distributions_function(trial)
    #         xs, _ = self._preprocess_trials([(trial.params, 0)], fit=False)
    #         x_pred, x_var = self._get_model_preds(xs)
    #         return self.acq_function(
    #             self.best_value,
    #             x_pred,
    #             x_var,
    #             noise=self._noise,
    #             random_state=self._rng.randint(0, 2 ** 16),
    #         )[0]

    #     last_best = None
    #     counter = 0
    #     early_stopping_delay = self.early_stopping_delay
    #     for i in range(self.n_ei_candidates):
    #         _run_trial(self._independent_study, opt_func)
    #         if not last_best or last_best < self._independent_study.best_value:
    #             last_best = self._independent_study.best_value
    #             counter = 0
    #         elif early_stopping_delay <= 0:
    #             counter += 1
    #         else:
    #             early_stopping_delay -= 1
    #         if counter >= self.early_stopping_patience:
    #             break

    #     print(self._independent_study.best_value)
    #     return self._independent_study.best_params

    def ask(
        self,
        trial: FrozenTrial,
    ) -> Dict[str, Any]:
        def opt_func(trial: Trial):
            self.distributions_function(trial)
            return 1

        trials = [
            _run_trial(self._independent_study, opt_func, self._rng)
            for _ in range(self.n_ei_candidates)
        ]
        xs, _ = self._preprocess_trials(trials, fit=False)
        x_pred, x_var = self._get_model_preds(xs)
        acq_values = self.acq_function(
            self.best_value,
            x_pred,
            x_var,
            noise=self._noise,
            random_state=self.random_state,
        )
        best_acq = np.argmax(acq_values)
        if not acq_values[best_acq]:
            best_acq = np.argmin(x_pred)
        return trials[best_acq][0]

    # def ask(
    #     self,
    #     trial: FrozenTrial,
    # ) -> Dict[str, Any]:
    #     def opt_func(trial: Trial):
    #         self.distributions_function(trial)
    #         return 1

    #     with redirect_stdout(io.StringIO()), redirect_stderr(
    #         io.StringIO()
    #     ), all_logging_disabled():
    #         self._independent_study.optimize(opt_func, n_trials=self.n_ei_candidates)
    #     trials = self._independent_study.get_trials(
    #         deepcopy=False, states=(TrialState.COMPLETE,)
    #     )
    #     xs, _ = self._preprocess_trials(trials, fit=False)
    #     x_pred, x_var = self._model.predict(xs)
    #     acq_values = self.acq_function(self.best_value, x_pred, x_var)
    #     best_acq = np.argmax(acq_values)
    #     return trials[best_acq].params

    def tell(
        self,
        complete_trials: List[FrozenTrial],
    ):
        self._complete_trials = complete_trials
        xs, ys = self._preprocess_trials(
            [(trial.params, trial.value) for trial in complete_trials], fit=True
        )
        self._model.fit(xs, ys)
        self._noise = mean_squared_error(ys, self._model.predict(xs)[0])

    def _preprocess_trials(self, trials: List[Tuple[Dict[str, Any], float]], fit: bool):
        x_nums = []
        x_cats = []
        ys = []

        for params, value in trials:
            x_num, x_cat, y = self._complete_to_observation(params, value)
            x_nums.append(x_num)
            x_cats.append(x_cat)
            ys.append(y)

        x_nums = pd.DataFrame(x_nums).astype(float)
        x_cats = pd.DataFrame(x_cats).astype("category")
        ys = pd.Series(ys).astype(float)
        if fit:
            ys = self._transform_y(ys)

        x_nums, x_cats = self._impute(x_nums, x_cats, fit)
        x_cats = self._encode(x_cats, ys, fit)
        return pd.concat((x_nums, x_cats), axis=1), ys

    def _scale_col(self, col: pd.Series) -> pd.Series:
        low, high = self.search_space_bounds[col.name]
        scale = 1 / (high - low)
        return scale * col - low * scale

    def _transform_y(self, ys: pd.Series) -> pd.Series:
        numpy_ys = ys.values.reshape(-1, 1)
        ys_std = ys.std()
        if ys.min() <= 0:
            ys_n = pd.Series(
                power_transform(numpy_ys / ys_std, method="yeo-johnson").flatten()
            )
        else:
            ys_n = pd.Series(
                power_transform(numpy_ys / ys_std, method="box-cox").flatten()
            )
        self.best_value = ys_n.min()
        return ys_n

    def _impute(
        self, x_nums: pd.DataFrame, x_cats: pd.DataFrame, fit: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not x_cats.empty:

            def handle_cat_column(col: pd.Series):
                return (
                    col.cat.rename_categories(lambda x: str(x))
                    .cat.add_categories("_missing_value")
                    .fillna("_missing_value")
                    .cat.remove_unused_categories()
                )

            x_cats = x_cats.apply(handle_cat_column)
        if not x_nums.empty:
            x_nums = x_nums.apply(self._scale_col).fillna(-1)

        return x_nums, x_cats

    def _encode(self, x_cats: pd.DataFrame, ys: pd.Series, fit: bool) -> pd.DataFrame:
        return (
            self._encoder.fit_transform(x_cats, ys)
            if fit
            else self._encoder.transform(x_cats)
        )


class CatBoostSamplerModel(RandomForestSamplerModel):
    def __init__(
        self,
        study: Study,
        distributions_function: Callable[[Trial], None],
        search_space: Dict[str, distributions.BaseDistribution],
        n_ei_candidates: int,
        best_value: float,
        independent_sampler: Union[BaseSampler, Type[BaseSampler]],
        *,
        acq_function: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = logei(0),
        n_estimators: int = 100,
        boostrap: bool = True,
        max_features: Union[str, float] = "auto",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        independent_sampler_kwargs: Optional[Dict[str, Any]] = None,
        early_stopping_patience: int = 50,
        early_stopping_delay: int = 128,
    ) -> None:
        self.study = study
        self.search_space = search_space
        self.search_space_bounds = self._get_search_space_bounds()
        self.acq_function = acq_function
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.bootstrap = boostrap
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        self.best_value = best_value
        self.n_ei_candidates = n_ei_candidates
        self.independent_sampler = independent_sampler(
            seed=random_state, **independent_sampler_kwargs
        )
        self.distributions_function = distributions_function
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delay = early_stopping_delay
        self.num_ensembles = 10

        self._rng = np.random.RandomState(self.random_state)
        self._model = CatBoostRegressorWithAutoCatFeatures(
            iterations=self.n_estimators,
            learning_rate=0.2,
            depth=10,
            loss_function="RMSEWithUncertainty",
            posterior_sampling=True,
            verbose=False,
            random_seed=random_state,
        )
        self._noise = 0

        with redirect_stdout(io.StringIO()), redirect_stderr(
            io.StringIO()
        ), all_logging_disabled():
            self._independent_study = optuna.create_study(
                storage=optuna.storages.InMemoryStorage(),
                sampler=self.independent_sampler,
                direction="maximize",
            )

    def tell(
        self,
        complete_trials: List[FrozenTrial],
    ):
        self._complete_trials = complete_trials
        xs, ys = self._preprocess_trials(
            [(trial.params, trial.value) for trial in complete_trials], fit=True
        )
        self._model.fit(xs, ys)
        self._noise = mean_squared_error(ys, self._model.predict(xs)[:, 0])

    def _get_model_preds(self, xs: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        preds = self._model.virtual_ensembles_predict(
            data=xs,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=self.num_ensembles,
        )
        x_pred = preds[:, 0]
        x_var = preds[:, 1] + preds[:, 2]
        return x_pred, x_var

    def _encode(self, x_cats: pd.DataFrame, ys: pd.Series, fit: bool) -> pd.DataFrame:
        return x_cats


class RandomForestSampler(BaseSampler):
    _model = RandomForestSamplerModel

    def __init__(
        self,
        *,
        distributions_function: Optional[Callable[[Trial], None]] = None,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 10,
        seed: Optional[int] = None,
        independent_sampler: Type[BaseSampler] = RandomSampler,
        random_fraction: float = 0.3,
        constant_liar: bool = False,
        warn_independent_sampling: bool = True,
        independent_sampler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert random_fraction < 1 and random_fraction >= 0
        # assert n_startup_trials >= 5
        self._distributions_function = distributions_function
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._independent_sampler_kwargs = independent_sampler_kwargs or {}
        self._independent_sampler_kwargs.pop("seed", None)
        self._independent_sampler = independent_sampler(
            seed=seed, **self._independent_sampler_kwargs
        )
        self._random_fraction = random_fraction
        self._constant_liar = constant_liar
        self._warn_independent_sampling = warn_independent_sampling
        self._search_space = _GroupDecomposedSearchSpace(True)
        self._search_space_group = None
        self._consider_pruned_trials = True
        self._worst_trial_value = float("-inf")
        self._best_trial_value = float("inf")

    def reseed_rng(self) -> None:
        self._rng = np.random.RandomState()
        self._independent_sampler.reseed_rng()

    def _get_trials(self, study: Study) -> Tuple[List[FrozenTrial], int]:
        complete_trials = []
        n_actually_completed_trials = 0
        for t in study.get_trials(deepcopy=False):
            if t.state == TrialState.COMPLETE:
                copied_t = t
                if study.direction == StudyDirection.MAXIMIZE:
                    copied_t = copy.deepcopy(t)
                    copied_t.value = -copied_t.value
                complete_trials.append(copied_t)
                n_actually_completed_trials += 1
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                copied_t = copy.deepcopy(t)
                copied_t.value = (
                    -value if study.direction == StudyDirection.MAXIMIZE else value
                )
                complete_trials.append(copied_t)
                n_actually_completed_trials += 1
            elif t.state == TrialState.RUNNING and self._constant_liar:
                copied_t = copy.deepcopy(t)
                copied_t.value = self._worst_trial_value
                complete_trials.append(copied_t)
        return complete_trials, n_actually_completed_trials

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:

        search_space = {}
        self._search_space_group = self._search_space.calculate(study)
        for sub_space in self._search_space_group.search_spaces:
            for name, distribution in sub_space.items():
                if distribution.single():
                    continue
                search_space[name] = distribution
        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:

        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        complete_trials, n = self._get_trials(study)

        if n < self._n_startup_trials:
            return {}

        if self._random_fraction and self._rng.uniform() < self._random_fraction:
            return {}

        assert self._distributions_function
        model = self._model(
            study=study,
            distributions_function=self._distributions_function,
            search_space=search_space,
            n_ei_candidates=self._n_ei_candidates,
            best_value=self._best_trial_value,
            independent_sampler=type(self._independent_sampler),
            random_state=self._rng.randint(0, 2 ** 16),
            independent_sampler_kwargs=self._independent_sampler_kwargs,
        )
        model.tell(complete_trials)
        self._last_model = model._model
        return model.ask(trial)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        self._raise_error_if_multi_objective(study)

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:

        logger = optuna.logging.get_logger(__name__)
        logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `{}` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `{}`, "
            "if this independent sampling is intended behavior.".format(
                param_name,
                trial.number,
                self.__class__.__name__,
                self._independent_sampler.__class__.__name__,
                self.__class__.__name__,
            )
        )

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if not values:
            return
        trial_value = values[0]
        if study.direction == StudyDirection.MAXIMIZE:
            trial_value = -trial_value
        if trial_value < self._best_trial_value:
            self._best_trial_value = trial_value
        if trial_value > self._worst_trial_value:
            self._worst_trial_value = trial_value


class CatBoostSampler(RandomForestSampler):
    _model = CatBoostSamplerModel
