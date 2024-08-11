# Soil Surface Roughness Change Detection

## Setup

This project run on Python using [Poetry](https://python-poetry.org/docs/) as a package manager. You can use `poetry` to install dependencies and run `poetry shell` and `make jupyter` to run Jupyter server and view the scripts.

## Run

### Pull Data

Data from Google Earth Engine are download using script in `dataloader.ipynb`. You need to have a Google account with a permission to use Google Earth Engine (see page on [Google Earth Engine](https://developers.google.com/earth-engine)). Also it is recommended to use Google Earth Engine with a service account, so it is more convinence to interact with the service programmatically (see (Service Account)[https://developers.google.com/earth-engine/guides/service_account]). After you have a service account, you can use it by specify the path and credential in `config.py`.

### Project Structure

- `data/` - contains annotated files such as Harrysfarm activities and field boundaries of the area of interest.
- `results/` - contains results from outlier detection scripts.
- `soil_roughness_change_detection`
    - `authentication.py` - contains helper function for authenticating with Google Earth Engine.
    - `experiment.py` - contains function for running experiments.
    - `outlier_detections.py` - contains outliers detection models and logics.
    - `preprocessor.py` - contains preprocessing logic for input data such as Sentinel-1 backscatter data, Sentinel-2 NDVI and GPM Precipitation data.
- `config.py` - contains config related to Google Could service account credential. Use `config_template.py` as a template to create this file.
- `Makefile` - contains a useful command to run Jupyter notebook in Poetry shell.
- `pyproject.toml`, `poetry.lock` - Poetry files for managing dependencies.

#### Scripts
- `dataloader.ipynb` - For download data from Google Earth Engine.
- `data_exploration.ipynb` - Contain Visualization for the download data.
- `experiment.ipynb` - Contain script for running the experiment.
- `evaluation.ipynb` - Contain script for evaluate the outlier detector with the testing set. It also contains visualization for the results.




