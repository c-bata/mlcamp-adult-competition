import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y, safe_sqr
from sklearn.feature_selection.base import SelectorMixin
from lightgbm import Booster

from typing import List, Optional, Callable, Any


def train_model(X_train, X_eval, y_train, y_eval):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
    }

    model = lgb.train(
        lgb_params, lgb_train,
        valid_sets=lgb_eval,
        early_stopping_rounds=20,
        num_boost_round=300,
        verbose_eval=False,
    )
    return model


def lightgbm_coef_func(booster: Booster) -> List[int]:
    return booster.feature_importance()


class RFE(SelectorMixin):
    """Recursive Feature Elimination for LightGBM

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.
    """
    def __init__(self,
                 train_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any] = train_model,
                 coef_func: Callable[[Any], List[int]] = lightgbm_coef_func,
                 n_features_to_select: Optional[int] = None,
                 step: int = 1,
                 verbose: int = 0,
                 random_state: Optional[int] = None,
                 test_size: Optional[float] = None):
        self.train_func = train_func
        self.coef_getter = coef_func
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.random_state = random_state
        self.test_size = test_size

    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        X, y = check_X_y(X, y, "csc", ensure_min_features=2, force_all_finite='allow-nan')
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        return self._fit(X_train, X_eval, y_train, y_eval)

    def _fit(self, X_train, X_eval, y_train, y_eval, step_score=None):
        # Initialization
        n_features = X_train.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            estimator = self.train_func(X_train[:, features], X_eval[:, features], y_train, y_eval)

            # Get coefs
            coefs = self.coef_getter(estimator)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"feature_importance()" method')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = self.train_func(X_train[:, features], X_eval[:, features], y_train, y_eval)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

    def _get_support_mask(self):
        return self.support_
