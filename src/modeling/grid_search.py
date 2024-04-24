import itertools
import warnings
from itertools import product

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.containers import GridSearchResult

warnings.filterwarnings("ignore")


def sarimax_grid_search(
    time_series: pd.Series,
    p_range: range,
    d_range: range,
    q_range: range,
    seasonal_p_range: range,
    seasonal_d_range: range,
    seasonal_q_range: range,
    seasonal_period: int = 12,
) -> GridSearchResult:
    parameters = product(p_range, d_range, q_range, seasonal_p_range, seasonal_d_range, seasonal_q_range)
    parameters_list = list(parameters)

    best_aic = float("inf")
    best_bic = float("inf")
    best_model = None
    best_params = None

    # Grid search
    for params in parameters_list:
        try:
            model = SARIMAX(
                time_series,
                order=(params[0], params[1], params[2]),
                seasonal_order=(params[3], params[4], params[5], seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=0)
            if results.aic < best_aic:
                best_aic = results.aic
                best_bic = results.bic
                best_model = results
                best_params = params
        except Exception as e:
            print(f"Model for params combination {params} couldn't be generated. Exception {str(e)}")
            continue

    params_names = ["p", "d", "q", "P", "D", "Q"]
    best_params_output = dict(zip(params_names, best_params))
    return GridSearchResult(best_aic=best_aic, best_bic=best_bic, best_params=best_params_output, best_model=best_model)


def arima_grid_search(time_series: pd.Series, p_range: range, q_range: range, d: int = 0) -> GridSearchResult:
    best_aic = float("inf")
    best_bic = float("inf")
    best_p = None
    best_q = None
    best_model = None

    for p, q in itertools.product(p_range, q_range):
        try:
            model = ARIMA(time_series, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_bic = results.bic
                best_p = p
                best_q = q
                best_model = results
        except Exception as e:
            print(f"Model for combination p: {p}, q: {q} couldn't be generated. Exception {str(e)}")
            continue

    return GridSearchResult(
        best_aic=best_aic, best_bic=best_bic, best_params={"best_p": best_p, "best_q": best_q}, best_model=best_model
    )
