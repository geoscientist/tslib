from datetime import date, timedelta
import pandas as pd
import numpy as np

def get_date_range(start, end):
    sdate = date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))   # start date
    edate = date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))   # end date
    delta = edate - sdate
    drange = []       

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        drange.append(day)

    return drange


def convert_atm_code(code: str):
    if code.isdigit():
        return str(int(code))
    else:
        return code


def fix_columns(df, date_fld):
    df[date_fld] = pd.to_datetime(df[date_fld])
    df["name"] = df["name"].astype("string")
    return df


def to_localtime(df, time_col, tz_col):
    df['local_time'] = df[time_col] + pd.to_timedelta((df[tz_col] - 3), unit='h')
    return df


def from_localtime(df, time_col, tz_col):
    df['date'] = df[time_col] - pd.to_timedelta((df[tz_col] - 3), unit='h')
    return df


def map_anomalies(data, y, yhat_min, threshold=0):
    data["anomaly"] = np.nan
    mask = np.bitwise_and(data[y] == 0, data[yhat_min] > threshold)
    data.loc[mask, "anomaly"] = 1
    return data
