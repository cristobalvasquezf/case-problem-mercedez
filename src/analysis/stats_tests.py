import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils import check_stationarity


def dickey_fuller_test(time_series: pd.Series):
    kpss_stat, p_value, _, num_observations, critical_values, _ = adfuller(time_series, autolag="AIC", maxlag=None)
    print(f"Dickey-fuller Statistic: {kpss_stat}")
    print(f"p-value: {p_value}")
    print(f"Critical Values: {critical_values}")
    check_stationarity(p_value)


def kpss_test(time_series: pd.Series):
    kpss_stat, p_value, _, critical_values = kpss(time_series, regression="c", nlags="auto")
    print(f"KPSS Statistic: {kpss_stat}")
    print(f"p-value: {p_value}")
    print(f"Critical Values: {critical_values}")
    check_stationarity(p_value)
