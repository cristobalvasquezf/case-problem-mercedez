import os
from typing import List, Dict, Tuple

import pandas as pd
from prefect import task, get_run_logger
from sktime.split import temporal_train_test_split

from src.analysis import kpss_test, dickey_fuller_test
from src.containers import GridSearchResult
from src.evaluation.metrics import compare_model_metrics
from src.modeling.forecaster import Forecaster
from src.modeling.grid_search import arima_grid_search, sarimax_grid_search
from src.postprocessing import Postprocessing
from src.preprocessing import Preprocessing


@task(name="preprocessing")
def preprocessing_task(data_path: str, apply_ln: bool = True) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Running preprocessing")
    preprocessing = Preprocessing(data_path=data_path, apply_ln=apply_ln)
    df = preprocessing.read_and_parse_data()
    logger.info("Preprocessing finished")
    return df


@task(name="data_analysis")
def data_analysis_task(time_series: pd.Series, save_image=True) -> None:
    logger = get_run_logger()
    logger.info(
        "Running data analysis: decomposing time serie in systematic parts and running stationary statistical tests, "
        "ACF and PACF plots"
    )
    # TODO: fix pipeline execution for tasks with plots
    # decompose_analysis(time_series=time_series, period=period, save_image=save_image)
    dickey_fuller_test(time_series=time_series)
    kpss_test(time_series=time_series)
    # plot_acf_and_pacf(time_series=time_series, num_lags=period, save_image=save_image)
    logger.info("Data analysis finished")


@task(name="split_data")
def split_data_task(df: pd.DataFrame, test_size: int = 12) -> Tuple[pd.Series, pd.Series]:
    """
    Split data into train and test sets
    :return: A tuple with (y_train, y_test)
    """
    logger = get_run_logger()
    logger.info("Splitting data for training")
    y_train, y_test = temporal_train_test_split(df["y"], test_size=test_size)
    logger.info("Splitting data finished")
    return y_train, y_test


@task(name="arima_grid_search")
def arima_grid_search_task(time_series: pd.Series, p_range: int = 5, q_range: int = 5, d: int = 0) -> GridSearchResult:
    logger = get_run_logger()
    logger.info("Running ARIMA grid search")
    grid_search_result = arima_grid_search(time_series, range(p_range), range(q_range), d)
    logger.info(f"Best params for ARIMA: {grid_search_result.best_params}")
    logger.info(f"ARIMA AIC: {grid_search_result.best_aic}")
    logger.info(f"ARIMA BIC: {grid_search_result.best_bic}")
    logger.info(grid_search_result.best_model.summary())
    logger.info("ARIMA grid search finished")
    return grid_search_result


@task(name="sarimax_grid_search")
def sarimax_grid_search_task(
        time_series: pd.Series,
        p_range: int = 3,
        d_range: int = 3,
        q_range: int = 2,
        seasonal_p_range: int = 2,
        seasonal_d_range: int = 2,
        seasonal_q_range: int = 2,
        seasonal_period: int = 12,
) -> GridSearchResult:
    logger = get_run_logger()
    logger.info("Running SARIMAX grid search")
    grid_search_result = sarimax_grid_search(
        time_series,
        range(p_range),
        range(d_range),
        range(q_range),
        range(seasonal_p_range),
        range(seasonal_d_range),
        range(seasonal_q_range),
        seasonal_period,
    )
    logger.info(f"Best params for SARIMAX: {grid_search_result.best_params}")
    logger.info(f"SARIMAX AIC: {grid_search_result.best_aic}")
    logger.info(f"SARIMAX BIC: {grid_search_result.best_bic}")
    logger.info(grid_search_result.best_model.summary())
    logger.info("SARIMAX grid search finished")
    return grid_search_result


@task(name="forecasting")
def forecasting_task(grid_search_output: GridSearchResult, model_name: str, y_test: pd.Series, steps: int = 12,
                     validate_model: bool = False) -> pd.DataFrame:
    """

    :param df: original dataframe
    :param grid_search_output:
    :param steps:
    :return:
    """
    logger = get_run_logger()
    logger.info(f"Running forecasting for next {steps} months with model {model_name}")
    postprocessing = Postprocessing(apply_ln=True)
    forecaster = Forecaster(grid_search_output.best_model, postprocessing)
    df_forecast = forecaster.forecast(steps=steps)
    logger.info("Forecasting finished")
    if validate_model:
        correlation, p_value_corr = forecaster.pred_correlation(y_pred=df_forecast['y'], y_test=y_test)
    return df_forecast


@task(name="Store forecasting results")
def store_forecasting_results_task(
        output_path: str,
        df_arima_forecast: pd.DataFrame,
        df_sarimax_forecast: pd.DataFrame,
        y_test: pd.Series,
        validate_model: bool = False,
) -> None:
    logger = get_run_logger()
    logger.info("Saving forecast results")
    arima_filename = "arima_validation_forecast.csv" if validate_model and y_test is not None else "arima_forecast.csv"
    sarimax_filename = (
        "sarimax_validation_forecast.csv" if validate_model and y_test is not None else "sarimax_forecast.csv"
    )
    df_arima_forecast.to_csv(os.path.join(output_path, arima_filename), index=False, header=True, columns=["date", "y"])
    df_sarimax_forecast.to_csv(
        os.path.join(output_path, sarimax_filename), index=False, header=True, columns=["date", "y"]
    )
    logger.info("Forecast results saved")


@task(name="metrics imputation")
def metrics_imputation_task(y_preds: Dict[str, List[float]], y_test: List[float]) -> Dict[str, float]:
    logger = get_run_logger()
    logger.info("Running metrics imputation")
    metrics_output = compare_model_metrics(y_preds, y_test)
    logger.info("Metrics imputation finished")
    return metrics_output
