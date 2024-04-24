# CASE PROBLEM

## Set up

Before running the project, you need to install poetry in your OS, then you can run the follow command.
For more details about poetry installation please refer to
the [official documentation](https://python-poetry.org/docs/).

```
poetry install
```

## Pipeline

The project is configured to create a virtualenv in the project folder called .venv this is done after running the
command poetry install
With the environment activated its possible to execute the pipeline with the follow command.

The pipeline consists in the follow stages:
1. Preprocessing of the data. There is a flag called apply_ln that indicates if natural logarithm should be applied to the
   data, this would help to stabilize variance and helps with exponential data behaviors seen during analysis and more. Some reference [here](https://juandelacalle.medium.com/best-tips-and-tricks-when-and-why-to-use-logarithmic-transformations-in-statistical-analysis-9f1d72e83cfc)
2. Data analysis with decomposition to see: trend, seasonality and residuals. Stationary tests are done also.
3. Split data only for model validation purposes. For train with full dataset set split_data to false.
4. Grid search over hyperparameters of ARIMA and SARIMAX models. All p,q,d and seasonal p,q,d are trained based on config.json file.
5. Forecasting is done with the best model found in the grid search.
6. The pipeline generates a csv file with the forecasting.
7. Model validation with the follow metrics: mean squared error, mean absolute error and mean absolute percentage error.

```
python src/pipeline.py -cf config.json
```

In this case -cf flag refers to a path where config.json is located. This file contains the configuration for the
pipeline.

The configuration file has the follow fields:

| Parameter        | Type    | Description                                                                                                     |
|:-----------------|:--------|:----------------------------------------------------------------------------------------------------------------|
| data_path        | String  | Path of train data in .csv format                                                                               |
| output_path      | String  | Folder path where to generate forecasting in .csv format                                                        |
| apply_ln         | Boolean | A boolean to indicate if natural logarithm should be applid to train data                                       |
| split_data       | Boolean | A flag used to split data in train and test                                                                     |
| test_size        | Integer | Number of months used in test data if applies                                                                   |
| validate_model   | Boolean | Flag used to validate the model. This flag needs data split in train and test                                   |
| p_range          | Integer | An integer used to do a grid search from 0 until the value indicated for p param in ARIMA model                 |
| q_range          | Integer | Same as q_range for q param in ARIMA model                                                                      |
| d                | Integer | Number of differentiation used by ARIMA model. In this case the data is likely stationary then 0 is recommended |
| d_range          | Integer | Range for d param used for grid search in SARIMAX model                                                         |
| seasonal_p_range | Integer | Range for seasonal p param used for grid search in SARIMAX model                                                |
| seasonal_d_range | Integer | Range for seasonal d param used for grid search in SARIMAX model                                                |
| seasonal_q_range | Integer | Range for seasonal q param used for grid search in SARIMAX model                                                |
| steps            | Integer | Number of steps to be forecasted                                                                                |
| period           | Integer | Number of periods used for analysis this data shows a yearly period then 12 is recommended.                     |

## Execution modes

The pipeline can be executed in two modes:

1. With model validation and split_data which is recommended for analysis and validation of the forecasted values
2. Without validation and split_data to use the entire dataset to train the model in order to have more data and improve
   the results against unseen data.

These two execution can be done by changing the configuration file. As follows:

1. For validation pipeline consider: split_data=true, test_size=12, validate_model=true. At project root refer to
   config_validation.json file
2. For full train pipeline: split_data=false, test_size=0, validate_model=false. At project root refer to config.json
   file

The pipeline was executed considering the two modes and the generated forecasting is located in the output folder (
data).

The outputs that must be considered for evaluation are:

1. data/arima_forecast.csv -> Baseline for comparison.
2. data/sarimax_forecast.csv -> This one should be better than the previous one.

## Folder in the project

1. images: there are two images that refer to time series decomposition analysis that can be found in jupyter notebook
   also.
2. notebooks: folder to store notebooks used to analyze the data and create the pipeline.
3. src: source code of the project.
4. data: folder to store the data used in the project and the forecasting generated by the pipeline.
5. doc: folder to store the documentation of the project and the reason of why python39 is used (statsmodels requires
   python39 or higher for the latest version).
6. tests: folder with source code for tests of the project.

## Analysis summary

The main analysis is done in the notebook called analysis.ipynb. Some highlights:

1. Time series decomposition is done analyzing trend, seasonality and residuals. With this analysis seasonality is found in the data and non-linear trend also.
2. Stationary tests are done to check if the data is stationary. The data is likely stationary based on the tests executed which is great because the models has this assumption.
3. Residual analysis was done finding that the residuals are likely normal distributed.
4. ACF and PACF analysis was done seeing peaks for lags in months 1, 5-7 and 12. Grid search is done anyways to find the best hyperparameters for the models.
5. The pipeline was executed with ARIMA and SARIMAX models. The best model found was SARIMAX with a better forecasting than ARIMA.
6. The model was validated with the follow metrics: mean squared error, mean absolute error and mean absolute percentage error. The results are in the notebook.
7. More models was tested from sktime lib and Prophet for validation but more work is needed.

## Tools used for coding

The tools used for coding are mainly poetry for dependencies management and black with flake8 for linting.
