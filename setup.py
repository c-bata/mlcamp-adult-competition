from setuptools import setup, find_packages

setup(
    name='automl-shibata',
    packages=find_packages(),
    entry_points={'console_scripts': ['automl = automl.cmd:cmd']},
)