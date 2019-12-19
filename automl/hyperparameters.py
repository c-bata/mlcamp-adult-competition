import os
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb

from lightgbm.basic import Booster
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import check_cv, train_test_split
from sklearn.utils import check_X_y, indexable
from typing import List

from automl import DATA_DIR
from automl.feature import generate_feature

train_x = pd.read_csv(os.path.join(DATA_DIR, 'train_x.csv'))
train_y = pd.read_csv(os.path.join(DATA_DIR, 'train_y.csv'), header=None)
test_x = pd.read_csv(os.path.join(DATA_DIR, 'test_x.csv'))


def train_model(X_train, X_eval, y_train, y_eval) -> Booster:
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


def calculate_auc(model: Booster, X_eval, y_eval) -> float:
    y_pred = model.predict(X_eval, num_iteration=model.best_iteration)
    return roc_auc_score(y_eval, y_pred)


def cross_val_score(model_trainer, X, y, cv) -> float:
    if cv <= 1:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.25, random_state=40 + 1)
        model = model_trainer(X_train, X_eval, y_train, y_eval)
        return calculate_auc(model, X_eval, y_eval)

    # pd.DataFrame to np.ndarray
    X, y = check_X_y(X, y, "csr", ensure_min_features=2,
                     force_all_finite='allow-nan')
    X, y = indexable(X, y)
    cv = check_cv(cv, y, True)

    scores: List[float] = []
    for idx_train, idx_eval in cv.split(X, y):
        X_train = X[idx_train]
        y_train = y[idx_train]
        X_eval = X[idx_eval]
        y_eval = y[idx_eval]
        model = model_trainer(X_train, X_eval, y_train, y_eval)
        score = calculate_auc(model, X_eval, y_eval)
        scores.append(score)

    return np.mean(scores)


def optimize(n_trials: int, synthesis: bool) -> optuna.Study:
    study = optuna.create_study(
        study_name='adult',
        storage='sqlite:///db.sqlite3',
        sampler=optuna.samplers.TPESampler(),
        direction='maximize',
        load_if_exists=True,
    )

    def objective(trial):
        n_features = trial.suggest_int("features", 4, 32)
        train_feature, test_feature = generate_feature(
            train_x, train_y, test_x, n_features, synthesis=synthesis)
        return cross_val_score(train_model, train_feature, train_y, cv=4)

    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=False)
    return study
