# from sklearn.base import clone, ClassifierMixin, BaseEstimator, TransformerMixin
# from sklearn.ensemble._base import _fit_single_estimator
# from sklearn.model_selection import train_test_split
# from sklearn.utils.validation import check_is_fitted
# from joblib import Parallel, delayed

# from ...compatibility.pandas import categorical_columns_to_int_categories


# class DESSplitter(ClassifierMixin, TransformerMixin, BaseEstimator):
#     def __init__(
#         self,
#         pool_classifiers,
#         des_estimator,
#         *,
#         DSEL_perc=0.25,
#         random_state=None,
#         n_jobs=None
#     ) -> None:
#         self.pool_classifiers = pool_classifiers
#         self.des_estimator = des_estimator
#         self.DSEL_perc = DSEL_perc
#         self.random_state = random_state
#         self.n_jobs = n_jobs

#     def fit(self, X, y, sample_weight=None):
#         X = categorical_columns_to_int_categories(X)
#         X_train, X_dsel, y_train, y_dsel = train_test_split(
#             X,
#             y,
#             random_state=self.random_state,
#             test_size=self.DSEL_perc,
#             stratify=y,
#         )
#         self.pool_classifiers_ = Parallel(n_jobs=self.n_jobs)(
#             delayed(_fit_single_estimator)(clone(est), X_train, y_train, sample_weight)
#             for est in self.pool_classifiers
#             if est != "drop"
#         )
#         self.des_estimator_ = clone(self.des_estimator)
#         self.des_estimator_.pool_classifiers = self.pool_classifiers_
#         self.des_estimator_.random_state = self.random_state
#         self.des_estimator_.fit(X_dsel, y_dsel)
#         return self

#     def predict(self, X):
#         X = categorical_columns_to_int_categories(X)
#         return self.des_estimator_.predict(X)

#     def predict_proba(self, X):
#         X = categorical_columns_to_int_categories(X)
#         return self.des_estimator_.predict_proba(X)

#     def score(self, X, y, sample_weight=None):
#         X = categorical_columns_to_int_categories(X)
#         return self.des_estimator_.score(X, y, sample_weight=sample_weight)

#     def __getattr__(self, item):
#         check_is_fitted(self)
#         return getattr(self.des_estimator_, item)
