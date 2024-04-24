import math

import pandas as pd


class Postprocessing:
    def __init__(self, apply_ln: bool = False):
        self._apply_ln = apply_ln

    @property
    def apply_ln(self):
        return self._apply_ln

    def postprocess(self, time_series: pd.Series) -> pd.Series:
        if self.apply_ln:
            return time_series.apply(lambda x: math.exp(x))
        return time_series

    def process_model_results(self):
        pass
