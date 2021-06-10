from pandas.api.types import is_integer_dtype
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split


def _auto_early_stopping_condition(early_stopping, X, y):
    if (
        early_stopping != "auto"
        and not is_integer_dtype(type(early_stopping))
        and not isinstance(early_stopping, bool)
    ):
        raise TypeError(
            f"early_stopping must be 'auto', an integer or a bool, got '{early_stopping}'' ({type(early_stopping)}!"
        )
    if early_stopping == "auto":
        return len(X) > 10000
    if is_integer_dtype(type(early_stopping)):
        return len(X) > early_stopping
    return bool(early_stopping)


class LGBMClassifierWithAutoEarlyStopping(LGBMClassifier):
    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=-1,
        silent=True,
        importance_type="split",
        early_stopping_condition="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        **kwargs,
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs,
        )
        self.early_stopping_condition = early_stopping_condition
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=False,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        if early_stopping_rounds is None and _auto_early_stopping_condition(
            self.early_stopping_condition, X, y
        ):
            try:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X,
                    y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=y,
                )
            except ValueError:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X,
                    y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    shuffle=True,
                )
            assert len(X_eval) > 0
            eval_set = (X_eval, y_eval)
            eval_metric = self.scoring if self.scoring != "loss" else None
            early_stopping_rounds = self.n_iter_no_change
        else:
            X_train = X
            y_train = y
        return super().fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )


class LGBMRegressorWithAutoEarlyStopping(LGBMRegressor):
    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=-1,
        silent=True,
        importance_type="split",
        early_stopping_condition="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        **kwargs,
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs,
        )
        self.early_stopping_condition = early_stopping_condition
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=False,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        if early_stopping_rounds is None and _auto_early_stopping_condition(
            self.early_stopping_condition, X, y
        ):
            X_train, X_eval, y_train, y_eval = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=True,
            )
            assert len(X_eval) > 0
            eval_set = (X_eval, y_eval)
            eval_metric = self.scoring if self.scoring != "loss" else None
            early_stopping_rounds = self.n_iter_no_change
        else:
            X_train = X
            y_train = y
        return super().fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )
