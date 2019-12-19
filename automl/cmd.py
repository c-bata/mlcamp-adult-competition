import functools
import sys
import click

from automl.model import train_and_predict
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
@click.option('--features', type=int, default=12,
              help='The number of features.')
@notify_if_catch_exception
def predict(features):
    """Train and predict using LightGBM."""

    auc = train_and_predict(n_features=features)
    print("AUC:", auc)


if __name__ == '__main__':
    cmd()
