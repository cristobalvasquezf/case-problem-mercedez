import argparse
import json
import time

from prefect import State, flow

from src.pipeline.tasks import (
    preprocessing_task,
    data_analysis_task,
    arima_grid_search_task,
    sarimax_grid_search_task,
    split_data_task,
    forecasting_task,
    metrics_imputation_task,
    store_forecasting_results_task,
)


@flow(name="model pipeline", log_prints=True)
def pipeline_flow(
    data_path: str,
    output_path: str,
    apply_ln: bool = True,
    split_data: bool = False,
    test_size: int = 12,
    validate_model: bool = True,
    p_range: int = 3,
    q_range: int = 3,
    d: int = 0,
    d_range: int = 3,
    seasonal_p_range: int = 2,
    seasonal_d_range: int = 2,
    seasonal_q_range: int = 2,
    period: int = 12,
    steps: int = 12,
):
    df = preprocessing_task(data_path=data_path, apply_ln=apply_ln)

    data_analysis_task(time_series=df["y"])

    y_train, y_test = split_data_task(df=df, test_size=test_size) if split_data else (df["y"], None)

    arima_grid_search_result = arima_grid_search_task(time_series=y_train, p_range=p_range, q_range=q_range, d=d)
    sarimax_grid_search_result = sarimax_grid_search_task(
        time_series=y_train,
        p_range=p_range,
        d_range=d_range,
        q_range=q_range,
        seasonal_p_range=seasonal_p_range,
        seasonal_d_range=seasonal_d_range,
        seasonal_q_range=seasonal_q_range,
    )

    df_arima_forecast = forecasting_task(grid_search_output=arima_grid_search_result, model_name="arima", steps=steps)
    df_sarimax_forecast = forecasting_task(
        grid_search_output=sarimax_grid_search_result, model_name="sarimax", steps=steps
    )

    store_forecasting_results_task(
        output_path=output_path,
        df_arima_forecast=df_arima_forecast,
        df_sarimax_forecast=df_sarimax_forecast,
        y_test=y_test,
        validate_model=validate_model,
    )

    if validate_model and y_test is not None:
        predictions = {
            "arima": df_arima_forecast["y"],
            "sarimax": df_sarimax_forecast["y"],
        }
        metrics_output = metrics_imputation_task(y_preds=predictions, y_test=y_test)
        print(metrics_output)


def run_pipeline(config_path: str) -> State:
    with open(config_path, "r") as f:
        config = json.load(f)

    result = pipeline_flow(
        data_path=config.get("data_path"),
        output_path=config.get("output_path"),
        apply_ln=config.get("apply_ln"),
        split_data=config.get("split_data"),
        test_size=config.get("test_size"),
        validate_model=config.get("validate_model"),
        p_range=config.get("p_range"),
        q_range=config.get("q_range"),
        d=config.get("d"),
        d_range=config.get("d_range"),
        seasonal_p_range=config.get("seasonal_p_range"),
        seasonal_d_range=config.get("seasonal_d_range"),
        seasonal_q_range=config.get("seasonal_q_range"),
        period=config.get("period"),
        steps=config.get("steps"),
        return_state=True,
    )

    return result


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-cf", "--configfile", type=str, help="Path of json file with configurations")
    args = args_parser.parse_args()

    start = time.perf_counter()
    run_pipeline(config_path=args.configfile)
    end = time.perf_counter()
