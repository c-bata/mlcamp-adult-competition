# ML camp adult competition

## How to run

1. Unarchived data and put it as `data/` directory on the root.
2. Run ``$ automl search`` to search best options using Optuna.
3. Run ``$ automl predict --features 32 --category-encoding label --fold 4`` to make submission file.

See `$ automl --help` for more details

## Command line options

```console
$ automl predict --help
Usage: automl predict [OPTIONS]

  Train and predict using LightGBM.

Options:
  --features INTEGER              The number of features.
  --featuretools / --no-featuretools
                                  Synthesis features using feature tools if
                                  True
  --category-encoding [label|ohe]
  --fold INTEGER                  The number of LightGBM model learned from N
                                  folded dataset.
  --save-model / --no-save-model  Save models in output directory, if true.
  --save-feature-importance / --no-save-feature-importance
                                  Save image of feature importance, if true.
  --help                          Show this message and exit.
```