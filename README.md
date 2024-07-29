
<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/repo_logo_dark.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="./assets/repo_logo_light.png" width='100%'>
  <img alt="Project logo" src="/assets/" width="100%">
</picture> -->

<br>

[![tests](https://github.com/schwallergroup/CycPepPerm/actions/workflows/tests.yml/badge.svg)](https://github.com/schwallergroup/CycPepPerm)
<!-- [![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)
[![PyPI](https://img.shields.io/pypi/v/CycPepPerm)](https://img.shields.io/pypi/v/CycPepPerm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/CycPepPerm)](https://img.shields.io/pypi/pyversions/CycPepPerm) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Cookiecutter template from @SchwallerGroup](https://img.shields.io/badge/Cookiecutter-schwallergroup-blue)](https://github.com/schwallergroup/liac-repo)
[![Learn more @SchwallerGroup](https://img.shields.io/badge/Learn%20%0Amore-schwallergroup-blue)](https://schwallergroup.github.io)


<h1 align="center">
  CycPepPerm
</h1>


<br>

Python package to predict membrane permeability of cyclic peptides.

## 👩‍💻 Installation

We provide the code as a [python package](https://pypi.org/project/cyc-pep-perm/), so the only thing you need is to install it. We recommend creating a new conda environment for that, which allows simple package management for a project. Follow these [instructions](https://docs.anaconda.com/free/anaconda/install/index.html) to install Anaconda. However, the package containing our code can also be installed without creating a project-specific environment. In that case, one just skips the first two lines of the following code:

```bash
conda create -n cyc-pep-perm python=3.10
conda activate cyc-pep-perm
pip install cyc-pep-perm==0.1.1
```

### 🛠️ For Developers
<details>
  <summary>See detailed installation instructions</summary>
The repository can be cloned from GitHub and installed with `pip` or `conda`. The code was built with Python 3.10 on Linux but other OS should work as well.

With conda:

```bash
$ git clone git+https://github.com/schwallergroup/CycPepPerm.git
$ cd CycPepPerm
$ conda env create -f environment.yml
$ conda activate cyc_pep_perm
$ pip install -e .
```

or with pip:

```bash
$ git clone git+https://github.com/schwallergroup/CycPepPerm.git
$ cd CycPepPerm
$ conda create -n cyc_pep_perm python=3.10
$ conda activate cyc_pep_perm
$ pip install -r requirements.txt
$ pip install -e .
```

If the options above did not work, please try from scratch:

```bash
$ git clone git+https://github.com/schwallergroup/CycPepPerm.git
$ cd CycPepPerm
$ conda create -c conda-forge -n cyc_pep_perm rdkit=2022.03.5 python=3.10
$ conda activate cyc_pep_perm
$ conda install -c conda-forge scikit-learn=1.0.2
$ conda install -c rdkit -c mordred-descriptor mordred
$ conda install -c conda-forge xgboost
$ conda install -c conda-forge seaborn
$ pip install shap
$ conda install -c conda-forge jupyterlab
$ pip isntall pandas-ods-reader
$ pip install -e .
```
</details>

## 🔥 Usage

> For some more examples on how to process data, train and evaluate the alogrithms, please consult the folder `notebooks/`. This folder also contains a notebook to perform polynomial fits as described in the paper.

All data paths in the following examples are taken from the hard-coded paths that work when one clones this repository. If you use the python package and download the data separately, please change the paths accordingly.

### Data preprocessing

Here we showcase how to handle the data as for our use-case. Some simple reformating is done (see also the notebook `notebooks/01_data_preparation.ipynb`) starting from `.ods` file with DataWarrior output (for data see [Data and Models](#data-and-models)).

```python
import os

from cyc_pep_perm.data.processing import DataProcessing

data_dir = "/path/to/data/folder" # ADAPT TO YOUR PATH!

# this can also be a .csv input
datapath = os.path.join(data_dir, "perm_random80_train_raw.ods")

# instantiate the class and make sure the columns match your inputed file - otherwise change arguments
dp = DataProcessing(datapath=datapath)

# make use of precomputed descriptors from DataWarrior
df = dp.read_data(filename="perm_random80_train_dw.csv")

# calculate Mordred deescripttors
df_mordred = dp.calc_mordred(filename="perm_random80_train_mordred.csv")
```

### Training

Make sure to have the data ready to be used. In order to make the hyperparameter search more extensive, please look into the respective python scripts (e.g. `src/cyc_pep_perm/models/randomforest.py`) and adjust the `PARAMS` dictionary.

```python
import os

from cyc_pep_perm.models.randomforest import RF

data_dir = "/path/to/data/folder" # ADAPT TO YOUR PATH!
train_data = os.path.join(data_dir, "perm_random80_train_dw.csv")
model_dir = "/path/to/model/folder" # ADAPT TO YOUR PATH!
rf_model_trained = os.path.join(model_dir, "rf_random_dw.pkl")

# instantiate class
rf_regressor = RF()

model = rf_regressor.train(
    datapath = train_data,
    savepath = rf_model_trained,
)

y_pred, rmse, r2 = rf_regressor.evaluate()
# will print training results, e.g.:
>>> RMSE: 8.45
>>> R2: 0.879
```

### Prediction

```python
import os

from cyc_pep_perm.models.randomforest import RF

data_dir = "/path/to/data/folder" # ADAPT TO YOUR PATH!
train_data = os.path.join(data_dir, "perm_random20_test_dw.csv")
model_dir = "/path/to/model/folder" # ADAPT TO YOUR PATH!
rf_model_trained = os.path.join(model_dir, "rf_random_dw.pkl")

# instantiate class
rf_regressor = RF()

# load trained model
rf_regressor.load(
    modelpath = rf_model_trained,
)

# data to predict on, e.g.:
df = pd.read_csv(train_data)
X = df.drop(columns=["SMILES"])

# predict
y_pred = rf_regressor.predict(X)
```

## Data and Models

All data required for reproducing the results in the paper are provided in the folder `data/`. Beware that due to the random nature of these models, the results might differ from the ones reported in the paper. The files found in `data/` are split into training and test data (randomly split 80/20) and with either the DataWarrior (dw) or the Mordred descriptors. The simple data processing can be found in the notebook `notebooks/01_data_preparation.ipynb`. The DataWarrior descriptors are computed with external software ([DataWarrior](https://openmolecules.org/datawarrior/)). The following files are provided:

- `data/perm_random20_test_dw.csv` - test data with DataWarrior descriptors
- `data/perm_random20_test_mordred.csv` - test data with Mordred descriptors
- `data/perm_random20_test_raw.ods` - test data before processing
- `data/perm_random80_train_dw.csv` - training data with DataWarrior descriptors
- `data/perm_random80_train_mordred.csv` - training data with Mordred descriptors
- `data/perm_random80_train_raw.ods` - training data before processing

The models are provided in the folder `models/` and can be loaded with the `load_model()` method of the respective class. The models provided are:

- `models/rf_random_dw.pkl` - Random Forest trained on DataWarrior descriptors
- `models/rf_random_mordred.pkl` - Random Forest trained on Mordred descriptors
- `models/xgb_random_dw.pkl` - XGBoost trained on DataWarrior descriptors
- `models/xgb_random_mordred.pkl` - XGBoost trained on Mordred descriptors

## ✅ Citation

```bibtex
@Misc{this_repo,
  author = { Rebecca M Neeser },
  title = { cyc_pep_perm - Python package to predict membrane permeability of cyclic peptides. },
  howpublished = {Github},
  year = {2023},
  url = {https://github.com/schwallergroup/CycPepPerm }
}
```


## 🛠️ For Developers


<details>
  <summary>See developer instructions</summary>

### 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/schwallergroup/CycPepPerm/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/schwallergroup/CycPepPerm/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/schwallergroup/CycPepPerm.git
$ cd CycPepPerm
$ tox -e docs
$ open docs/build/html/index.html
```

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/cyc_pep_perm/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.
</details>
