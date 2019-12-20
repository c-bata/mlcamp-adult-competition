import os

import numpy as np
import pandas as pd

from lightgbm import Booster
from optuna.integration import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import check_cv
from sklearn.utils import indexable
from typing import List

from automl import OUTPUT_DIR


class EnsembleEstimator:
    def __init__(self, train_func, fold):
        assert fold > 1

        self.train_func = train_func
        self.fold: int = fold
        self.models: List[Booster] = []

    def fit(self, X, y):
        X, y = indexable(X, y)
        cv = check_cv(self.fold, y, classifier=True)
        for idx_train, idx_eval in cv.split(X, y):
            X_train = X.loc[idx_train]
            y_train = y.loc[idx_train]
            X_eval = X.loc[idx_eval]
            y_eval = y.loc[idx_eval]
            self.models.append(self.train_func(X_train, X_eval, y_train, y_eval))

    def predict(self, X):
        assert len(self.models) > 1

        predictions = [model.predict(X, num_iteration=model.best_iteration)
                       for model in self.models]
        return np.mean(predictions, axis=0) / len(predictions)


def train(X_train, X_eval, y_train, y_eval) -> Booster:
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
    }
    model = lgb.train(lgb_params, lgb_train,
                      valid_sets=lgb_eval, early_stopping_rounds=20,
                      num_boost_round=300, verbose_eval=False)
    return model


def train_and_predict(
        train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame,
        fold: int = 1, save_model: bool = False, save_feature_importance: bool = False) -> float:
    if fold > 1:
        model = EnsembleEstimator(train, fold)
        model.fit(train_x, train_y)
    else:
        X_train, X_eval, y_train, y_eval = train_test_split(train_x, train_y, test_size=0.25, random_state=40)
        model = train(X_train, X_eval, y_train, y_eval)
    prediction = model.predict(test_x)

    models = model.models if isinstance(model, EnsembleEstimator) else [model]
    if save_model:
        for i, m in enumerate(models):
            m.save_model(os.path.join(OUTPUT_DIR, f'lightgbm-{fold}-fold-{i}.txt'))
    if save_feature_importance:
        from lightgbm.plotting import plot_importance
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(model), figsize=(8, 4 * len(models),))
        for i, m in enumerate(models):
            plot_importance(booster=m, ax=axes[i])
        plt.savefig(os.path.join(OUTPUT_DIR, 'lightgbm-feature-importance.png'))
    return prediction
