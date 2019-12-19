import pandas as pd

from optuna.integration import lightgbm as lgb
from sklearn.model_selection import train_test_split


def train_and_predict(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame) -> float:
    X_train, X_eval, y_train, y_eval = train_test_split(train_x, train_y, test_size=0.25, random_state=40)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
    }
    model = lgb.train(lgb_params, lgb_train,
                      valid_sets=lgb_eval, early_stopping_rounds=20,
                      num_boost_round=300, verbose_eval=False)
    return model.predict(test_x)
