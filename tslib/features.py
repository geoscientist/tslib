import pandas as pd
import json
from .utils import *

def extract_features(data, time_col, lags, window):
        lag_max = lags.max()
        extra_data = get_days_features(data, time_col)
        tr_mode_feats = calc_tr_mode(data, window)
        lags = get_lag_features(data, lags, "name")
        data = pd.concat([lags, extra_data.astype(float), tr_mode_feats], axis=1)
        print("all data - ", data.shape)
        data = data.dropna()
        print("data after remove Nan - ", data.shape)
        y = data.tr_count
        data = data.drop(columns = "tr_count")

        return lag_max, data, y


def calc_tr_mode(df, w):
    '''
    params:
        df - pandas.DataFrame
        w - resampling window in format '4H'
    return: pandas.DataFrame
    '''
    d = df[["name", "local_time", "tr_count"]].groupby("name").resample(rule=w, on="local_time").mean()
    d = d.reset_index()
    d = d.groupby(["name", d.local_time.dt.hour]).mean().reset_index()
    d = pd.pivot(d, index="name", columns="local_time").reset_index()
    d = df.merge(d, on="name")
    return d.iloc[:, 3:]


def get_object_stat(df, agg_func):
        d = df[["name", "local_time", "tr_count"]].groupby("name").resample(rule="1D", on="local_time").sum()
        d = d.reset_index()
        d = d.groupby("name").mean().reset_index()
        d["tr_day"] = d["tr_count"]
        d = d.drop(columns="tr_count")
        d = df.groupby("name").agg(agg_func).reset_index()
        d = df.merge(d, on="name")
        return d.iloc[:, 3]


def get_days_features(df, time_col):
    data = pd.DataFrame(index=df.index)
    data["friday"] = df[time_col].dt.dayofweek.values == 4
    data["saturday"] = df[time_col].dt.dayofweek.values == 5
    data["sunday"] = df[time_col].dt.dayofweek.values == 6
    data["working"] = np.bitwise_and(df[time_col].dt.dayofweek.values >= 0, df[time_col].dt.dayofweek.values <= 3)
    data["pay_day"] = np.bitwise_or(df[time_col].dt.day.values == 5, df[time_col].dt.day.values == 20)
    data["after_pay_day"] = np.bitwise_or(df[time_col].dt.day.values == 6, df[time_col].dt.day.values == 21)
    return data


def get_lag_features(data, lag, agg_field):
    for l in lag:
        data[f"lag_{l}"] = data.groupby(agg_field)['tr_count'].shift(l)
    return data



def add_service_mode_feature(df, atm_registry):
    #atm_skto = atm_skto[["name", "Type", "Timezone", "Vendor", "Model"]]
    atm_registry = atm_registry[["name", "service_time"]]
    atm_registry['name'] = atm_registry['name'].astype("string")
    df = df.merge(atm_registry, on="name")
    service_mode_map = json.load(open("./data/service_mode.json"))
    df['service_mode'] = df.Service_time.apply(lambda x: service_mode_map[x])
    service_mode = pd.get_dummies(df.service_mode)
    df = df.drop(columns=["service_time", "service_mode"])
    df = df.join(service_mode)
    return df