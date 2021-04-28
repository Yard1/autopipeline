import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoostClassifierWithAutoCatFeatures(CatBoostClassifier):
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
            X,
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


class CatBoostRegressorWithAutoCatFeatures(CatBoostRegressor):
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
            X,
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
