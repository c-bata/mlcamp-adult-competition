import functools
import os
import sys
import click
import pandas as pd

from optuna.integration import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from automl import OUTPUT_DIR, DATA_DIR
from automl.notification import notify_slack


def notify_if_catch_exception(func):
    @functools.wraps(func)
    def callback(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            notify_slack(f"""*ERROR*
        Command: {sys.argv}
        Message: {str(e)}""")
            raise e
    return callback


@click.group()
def cmd():
    """Command line interface for machine learning camp competition."""
    pass


@cmd.command()
@click.option('--fold', type=int, default=4,
              help='The number of LightGBM model learned from N folded dataset.')
@notify_if_catch_exception
def predict(fold):
    """Train and predict using LightGBM."""

    click.echo(f'The number of fold: {fold}')

    # Load dataset
    train_x = pd.read_csv(os.path.join(DATA_DIR, 'train_x.csv'))
    train_y = pd.read_csv(os.path.join(DATA_DIR, 'train_y.csv'), header=None)
    test_x = pd.read_csv(os.path.join(DATA_DIR, 'test_x.csv'))

    train_x = train_x.fillna(value='?')
    train_x['sex'] = train_x['sex'] == 'Male'
    test_x = test_x.fillna(value='?')
    test_x['sex'] = test_x['sex'] == 'Male'

    # categorical encoding
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship',
        'race', 'capital-gain', 'native-country']
    features = train_x.append(test_x, ignore_index=True)
    for column in categorical_columns:
        le = preprocessing.LabelEncoder()
        le.fit(features[column])
        train_x[column] = pd.Series(le.transform(train_x[column])).astype('category')
        test_x[column] = pd.Series(le.transform(test_x[column])).astype('category')

    # train model
    X_train, X_eval, y_train, y_eval = train_test_split(train_x, train_y, test_size=0.25, random_state=40)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
    }

    model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=20, num_boost_round=300,
                      verbose_eval=False)

    submit_pred = model.predict(test_x)
    pd.Series(submit_pred).to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False, header=False)

    test_y = pd.read_csv(os.path.join(DATA_DIR, 'test_y.csv'), header=None)
    final_score = roc_auc_score(test_y, submit_pred)
    print("AUC:", final_score)


if __name__ == '__main__':
    cmd()
