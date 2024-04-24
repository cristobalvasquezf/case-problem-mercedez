from typing import Tuple

import pandas as pd
from scipy import stats

from src.postprocessing import Postprocessing


class Forecaster:
    def __init__(self, model: object, postprocessing: Postprocessing):
        self.model = model
        self.postprocessing = postprocessing

    def _parse_forecast_output(self, forecast_output: pd.Series) -> pd.DataFrame:
        print(forecast_output.summary_frame())
        df_output = pd.DataFrame(
            self.postprocessing.postprocess(forecast_output.predicted_mean)
            if self.postprocessing.apply_ln
            else forecast_output.predicted_mean
        )
        df_output["date"] = df_output.index
        df_output["date"] = df_output.date.apply(lambda x: x.strftime("%d.%m.%y"))
        df_output.rename(columns={"predicted_mean": "y"}, inplace=True)
        df_output[["date", "y"]].reset_index()
        return df_output

    def pred_correlation(self, y_pred: list, y_test: list) -> Tuple[float, float]:
        """

        :param y_pred: prediction done by the model
        :param y_test: test data to compare against
        :return: a tuple with the spearman correlation and pvalue
        """
        result = stats.spearmanr(y_pred, y_test)
        return result.statistic, result.pvalue

    def forecast(self, steps: int = 12) -> pd.DataFrame:
        df = self.model.get_forecast(steps=steps)
        df_forecast = self._parse_forecast_output(df)
        return df_forecast
