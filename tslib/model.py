import os
from datetime import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import fbprophet
from .features import extract_features
from joblib import dump, load


class ModelZoo():

    def __init__(self):
            self.all_model = load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
             "./checkpoints/tr_all_gbm_03.joblib"))
            self.cashIn_model = load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
             "./checkpoints/tr_ci_gbm_03.joblib"))
            self.cashOut_model = load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
             "./checkpoints/tr_co_gbm_03.joblib"))
            self.nfc_model = load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
             "./checkpoints/tr_nfc_gbm_03.joblib"))

class ARModel:
    sigma = 0.0

    def __init__(self, lags, target="tr_count", timestamp="timestamp", lower_depth=6, upper_depth=8):
        self.timestamp_col = timestamp
        self.lags = lags
        self.target_col = target
        self.search = GridSearchCV(
            estimator=GradientBoostingRegressor(n_estimators=100, random_state=0),
            cv=4,
            param_grid={"max_depth": range(lower_depth, upper_depth),
                        "n_estimators": [200],
                        "min_samples_leaf": [2]}
        )
        self.estimator = None

    
    def fit(self, x, y):
        self.search.fit(x, y)
        self.estimator = self.search.best_estimator_
        print(f"best params: {self.search.best_params_}")
        print(f"best score: {self.search.best_score_}")
        yp = self.estimator.predict(x)
        s = np.sum((y - yp) ** 2)
        n = y.shape[0]
        self.sigma = np.sqrt(s / (n - 1))
        print(s, n, self.sigma)

    def predict(self, x):
        n = x.shape[0]
        print(x.columns)
        yp = self.estimator.predict(x)
        result = x.copy()
        result.loc[:, "yhat"] = np.nan
        result.loc[:, "yhat_lower"] = np.nan
        result.loc[:, "yhat_upper"] = np.nan
        result["yhat"] = yp
        result["yhat_lower"] = yp - 1.50 * self.sigma
        result["yhat_upper"] = yp + 1.50 * self.sigma

        return result.reset_index().rename(columns={self.timestamp_col: "ds"})

    # TODO refactor this func
    def score(self, data):
        result = self.predict(data)
        tmp = result.dropna(how="any")

        return {
            "R2": r2_score(tmp[self.target_col], tmp["yhat"]),
            "MSE": mean_squared_error(tmp[self.target_col], tmp["yhat"]),
            "MAE": mean_absolute_error(tmp[self.target_col], tmp["yhat"])
        }

    def save(self, fname):
        dump(self, fname)

    def load(self, fname):
        self = load(fname)
        return self



# TODO move this settings to another module
holidays = pd.DataFrame({
    "holiday": ["1"],
    "ds": pd.to_datetime(["2019-10-20"]),
    "lower_window": 0,
    "upper_window": 0,
})


class ProphetModel:

    def __init__(self, timestamp="timestamp", target="tr_count", use_pay_day=True):
        self.timestamp_col = timestamp
        self.target_col = target
        self.model = fbprophet.Prophet(
            yearly_seasonality=False,
            daily_seasonality=False
        )
        # self.model.add_seasonality(name="weekly", period=7, fourier_order=1, prior_scale=10, mode="multiplicative")
        self.model.add_seasonality(
            name="friday",
            period=1,
            fourier_order=8,
            prior_scale=10,
            mode="multiplicative",
            condition_name="friday"
        )
        self.model.add_seasonality(
            name="saturday",
            period=1,
            fourier_order=8,
            prior_scale=10,
            mode="multiplicative",
            condition_name="saturday"
        )
        self.model.add_seasonality(
            name="sunday",
            period=1,
            fourier_order=8,
            prior_scale=10,
            mode="multiplicative",
            condition_name="sunday"
        )
        self.model.add_seasonality(
            name="working",
            period=1,
            fourier_order=8,
            prior_scale=10,
            mode="multiplicative",
            condition_name="working"
        )
        if use_pay_day:
            self.model.add_seasonality(
                name="pay_day",
                period=1,
                fourier_order=8,
                prior_scale=10,
                mode="multiplicative",
                condition_name="pay_day"
            )
            self.model.add_seasonality(
                name="after_pay_day",
                period=1,
                fourier_order=8,
                prior_scale=10,
                mode="multiplicative",
                condition_name="after_pay_day"
            )
        self.model.add_country_holidays(country_name='RU')

    def __convert(self, data):
        # TODO refactor feature engineering from model
        tmp = data.copy()
        tmp["friday"] = data.index.dayofweek.values == 4
        tmp["saturday"] = data.index.dayofweek.values == 5
        tmp["sunday"] = data.index.dayofweek.values == 6
        tmp["working"] = np.bitwise_and(data.index.dayofweek.values >= 0, data.index.dayofweek.values <= 3)
        tmp["pay_day"] = np.bitwise_or(data.index.day.values == 5, data.index.day.values == 20)
        tmp["after_pay_day"] = np.bitwise_or(data.index.day.values == 6, data.index.day.values == 21)
        return tmp.reset_index().rename(columns={self.timestamp_col: "ds", self.target_col: "y"})

    def fit(self, data):
        self.model.fit(self.__convert(data))

    def predict(self, data):
        return self.model.predict(self.__convert(data))

    def score(self, data):
        train_data = self.__convert(data)
        result = self.model.predict(train_data)
        return {
            "R2": r2_score(data[self.target_col], result["yhat"]),
            "MSE": mean_squared_error(data[self.target_col], result["yhat"]),
            "MAE": mean_absolute_error(data[self.target_col], result["yhat"]),
        }
