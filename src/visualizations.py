import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal


def plot_decomposed_analysis(decomposed_data: seasonal.DecomposeResult, title: str, save_image: bool = False) -> None:
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.subplot(411)
    plt.plot(decomposed_data.observed, label="Observed")
    plt.legend(loc="upper left")
    plt.subplot(412)
    plt.plot(decomposed_data.trend, label="Trend")
    plt.legend(loc="upper left")
    plt.subplot(413)
    plt.plot(decomposed_data.seasonal, label="Seasonality")
    plt.legend(loc="upper left")
    plt.subplot(414)
    plt.plot(decomposed_data.resid, label="Residuals")
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save_image:
        plt.savefig(f"decomposed_{title}.png")
        plt.close()
    else:
        plt.show()


def plot_acf_and_pacf(time_series: pd.Series, num_lags: int = 12, save_image: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(time_series, lags=num_lags, ax=axes[0])
    plot_pacf(time_series, lags=num_lags, ax=axes[1], method="ywm")
    if save_image:
        plt.savefig("acf_and_pacf.png")
        plt.close()
    else:
        plt.show()
