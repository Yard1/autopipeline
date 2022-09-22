import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from pandas.api.types import is_categorical_dtype


def categorical_column_to_int(col):
    if is_categorical_dtype(col.dtype):
        col = col.copy()
        col = col.cat.codes
    return col


class CatBoostClassifierWithAutoCatFeatures(
    CatBoostClassifier, ClassifierMixin, BaseEstimator
):
    def get_params(self, deep=True):
        r = super().get_params(deep=deep)
        r["auto_class_weights"] = r.get("auto_class_weights", None)
        return r

    def fit(
        self,
        X,
        y=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        sample_weight=None,
        baseline=None,
        use_best_model=None,
        eval_set=None,
        verbose=None,
        logging_level=None,
        plot=False,
        column_description=None,
        verbose_eval=None,
        metric_period=None,
        silent=None,
        early_stopping_rounds=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        init_model=None,
    ):
        if isinstance(X, pd.DataFrame) and not cat_features:
            cat_features = list(X.select_dtypes(include="category").columns)
        return super().fit(
            X.apply(categorical_column_to_int),
            y=y,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
            sample_weight=sample_weight,
            baseline=baseline,
            use_best_model=use_best_model,
            eval_set=eval_set,
            verbose=verbose,
            logging_level=logging_level,
            plot=plot,
            column_description=column_description,
            verbose_eval=verbose_eval,
            metric_period=metric_period,
            silent=silent,
            early_stopping_rounds=early_stopping_rounds,
            save_snapshot=save_snapshot,
            snapshot_file=snapshot_file,
            snapshot_interval=snapshot_interval,
            init_model=init_model,
        )


class CatBoostRegressorWithAutoCatFeatures(
    CatBoostRegressor, RegressorMixin, BaseEstimator
):
    def __repr__(self, N_CHAR_MAX=700) -> str:
        return BaseEstimator.__repr__(self, N_CHAR_MAX=N_CHAR_MAX)

    def fit(
        self,
        X,
        y=None,
        cat_features=None,
        sample_weight=None,
        baseline=None,
        use_best_model=None,
        eval_set=None,
        verbose=None,
        logging_level=None,
        plot=False,
        column_description=None,
        verbose_eval=None,
        metric_period=None,
        silent=None,
        early_stopping_rounds=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        init_model=None,
    ):
        if isinstance(X, pd.DataFrame) and not cat_features:
            cat_features = list(X.select_dtypes(include="category").columns)
        return super().fit(
            X.apply(categorical_column_to_int),
            y=y,
            cat_features=cat_features,
            sample_weight=sample_weight,
            baseline=baseline,
            use_best_model=use_best_model,
            eval_set=eval_set,
            verbose=verbose,
            logging_level=logging_level,
            plot=plot,
            column_description=column_description,
            verbose_eval=verbose_eval,
            metric_period=metric_period,
            silent=silent,
            early_stopping_rounds=early_stopping_rounds,
            save_snapshot=save_snapshot,
            snapshot_file=snapshot_file,
            snapshot_interval=snapshot_interval,
            init_model=init_model,
        )
