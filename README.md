# ML camp adult competition

## How to run

1. Unarchived data and put it as `data/` directory on the root.
2. Run ``$ automl search`` to search best options using Optuna.
3. Run ``$ automl predict --features 32 --category-encoding label --featuretools --fold 4 --save-feature-importances`` to make submission file.

See `$ automl --help` for more details
