import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa import seasonal

from src.visualizations import plot_decomposed_analysis


def residual_analysis(df: pd.DataFrame):
    for lag, stats in df.dropna().iterrows():
        # [0] is the stat and [1] the p_value
        if stats[1] > 0.05:
            print(f"There is autocorrelation in the residuals for lags {lag}")
        else:
            print(f"There is not autocorrelation in the residuals for lags {lag}")


def decompose_analysis(time_series: pd.Series, period: int = 12, save_image: bool = True):
    model_types = ["additive", "multiplicative"]
    for model_type in model_types:
        decomposed_data = seasonal.seasonal_decompose(time_series, model=model_type, period=period)
        plot_decomposed_analysis(decomposed_data, f"{model_type} decomposition", save_image=save_image)

        residuals = decomposed_data.resid
        residuals = residuals.dropna()

        print(f"Residuals analysis for {model_type} model")
        df = acorr_ljungbox(residuals, lags=list(range(period)), return_df=True)
        residual_analysis(df)
