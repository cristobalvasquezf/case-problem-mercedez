import math

import pandas as pd
import copy


class Preprocessing:
    def __init__(self, data_path: str, apply_ln: bool = False):
        """

        :param data_path: train data path
        :param apply_ln: a flag to indicate if we should apply log transformation to the target variable
        """
        self.data_path = data_path
        self.apply_ln = apply_ln

    def read_and_parse_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, header=0, names=["date", "y"])
        df.index = pd.to_datetime(df["date"], format="%d.%m.%y")
        df = df.drop(columns=["date"])
        if self.apply_ln:
            df["y"] = df["y"].apply(lambda x: math.log(x))
        return df.dropna()

    @staticmethod
    def create_lagged_features(df: pd.DataFrame, number_of_lags: int) -> pd.DataFrame:
        new_df = copy.deepcopy(df)
        for n in range(1, number_of_lags + 1):
            new_df[f"y_lag_{n}"] = new_df["y"].shift(n)
        return new_df.dropna()
