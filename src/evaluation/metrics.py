from typing import List, Dict

from sktime.performance_metrics.forecasting import mean_absolute_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_squared_error


def compare_model_metrics(y_preds: Dict[str, List[float]], y_test: List[float]) -> Dict[str, float]:
    metrics_output = {}
    for model_name, y_hat in y_preds.items():
        print(f"Imputing metrics for model {model_name}")
        metrics_output[f"mae_{model_name}"] = mean_absolute_error(y_test, y_hat)
        metrics_output[f"mse_{model_name}"] = mean_squared_error(y_test, y_hat)
        metrics_output[f"mape_{model_name}"] = mean_absolute_percentage_error(y_test, y_hat)

    print(f"The best metric is {min(metrics_output, key=metrics_output.get)}")
    return metrics_output
