import functools
import os
import pandas as pd
import sys
import click
from sklearn.metrics import roc_auc_score

from automl import DATA_DIR, OUTPUT_DIR
from automl.feature import generate_feature
from automl.model import train_and_predict
from automl.notification import notify_slack
from automl.hyperparameters import optimize


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
@click.option('--trials', type=int, default=12,
              help='The number of trials.')
@click.option('--synthesis/--no-synthesis', default=False,
              help='Synthesis feature if true')
@notify_if_catch_exception
def search(trials, synthesis):
    """Search hyperparameters."""
    study = optimize(trials, synthesis)
    click.echo(f"best_trial: {study.best_trial}")
    click.echo(f"$ optuna dashboard --study adult --storage sqlite:///db.sqlite3")


@cmd.command()
@click.option('--features', type=int, default=None,
              help='The number of features.')
@click.option('--synthesis/--no-synthesis', default=False,
              help='Synthesis feature if true')
@notify_if_catch_exception
def predict(features, synthesis):
    """Train and predict using LightGBM."""
    train_x = pd.read_csv(os.path.join(DATA_DIR, 'train_x.csv'))
    train_y = pd.read_csv(os.path.join(DATA_DIR, 'train_y.csv'), header=None)
    test_x = pd.read_csv(os.path.join(DATA_DIR, 'test_x.csv'))

    train_feature, test_feature = generate_feature(
        train_x, train_y, test_x, features, synthesis=synthesis)

    predictions = train_and_predict(train_feature, train_y, test_feature)
    pd.Series(predictions).to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False, header=False)

    test_y = pd.read_csv(os.path.join(DATA_DIR, 'test_y.csv'), header=None)
    auc = roc_auc_score(test_y, predictions)
    click.echo(f"AUC: {auc}")


if __name__ == '__main__':
    cmd()
