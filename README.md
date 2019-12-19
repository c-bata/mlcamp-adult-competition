# airtrack-competition

## How to run

1. Unarchived data and put it as `data/` directory on the root.
2. Run ``$ automl predict`` to make submission file.

See `$ automl --help` for more details

## Docker

```console
$ docker build -t airtrack-competition .
$ docker run -it --rm -v output_docker:/usr/src/output/ airtrack-competition automl train --fold4
```
