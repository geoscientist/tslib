from matplotlib import pyplot as plt
import sqlite3
import bootstrap as bs
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, RANSACRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import fbprophet
from datetime import datetime
from .model import ARModel
from sklearn.metrics import r2_score


def map_anomalies(data):

    data["anomalies"] = np.nan
    mask = np.bitwise_and(data["tr_count"] == 0, data["yhat_lower"] > 0)
    data.loc[mask, "anomalies"] = data.loc[mask, "tr_count"]

    return data


def adjust_transaction(tr):
    if np.isfinite(tr):
        return np.round(tr) if tr > 0 else 0
    else:
        return tr


def get_report(result):
    result = result.set_index("ds")
    week = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    report = pd.DataFrame()

    report["date"] = result.index.date
    report["weekday"] = result.index.dayofweek.map(lambda d: week[d])
    report["time"] = result.index.time
    report["real_transactions"] = result["tr_count"].values
    report["expected_lower"] = result["yhat_lower"].apply(adjust_transaction).values
    report["expected_middle"] = result["yhat"].apply(adjust_transaction).values
    report["expected_upper"] = result["yhat_upper"].apply(adjust_transaction).values
    report["abnormal"] = result["anomalies"].apply(lambda x: 1 if np.isfinite(x) else 0).values

    return report


connection = sqlite3.connect(f"{bs.path_to_db}/trans.db")

atm_id = "397901"

transactions = pd.read_sql(
    f"SELECT timestamp, sum(tr_count) as tr_count FROM transactions "
    f"WHERE name = '{atm_id}' and type != 999 GROUP BY timestamp ",
    connection,
    parse_dates="timestamp"
)

timezone_shift = pd.read_sql(
    f"SELECT `Часовой пояс` FROM info WHERE ID = '{atm_id}'",
    connection
).values[0, 0].astype(int)

service_start = pd.read_sql(
    f"select `ID УС` as atm_id, `Дата начала операций` as start_date from info2 where atm_id = '{atm_id}' ",
    connection,
    index_col="atm_id"
)

delta = pd.Timedelta("4W")

t2 = pd.Timestamp("2020-05-31")
if atm_id in service_start.index:
    t1 = service_start.loc[atm_id, "start_date"]
else:
    t1 = None
if t1 is None:
    t1 = t2 - delta

freq = "1H"


transactions["timestamp"] = transactions["timestamp"].apply(lambda t: t + pd.Timedelta(hours=timezone_shift - bs.transactions_timezone))
transactions.set_index(["timestamp"], inplace=True)
transactions_hourly = transactions.resample(freq).sum().fillna(0).loc[t1: t2]

print("average transactions per day: ", transactions_hourly.resample("1D").sum().mean())

transactions_hourly.plot(title="hourly plot")
plt.show()

data = transactions_hourly["tr_count"].values
t = transactions_hourly.index.values
plt.plot(t, data)

# Prophet prediction

model2 = Model(use_pay_day=True)

t_start = datetime.now()
model2.fit(transactions_hourly)
res = model2.predict(transactions_hourly)
score = model2.score(transactions_hourly)
print(datetime.now() - t_start)

res["tr_count"] = data
res = map_anomalies(res)
proph = res["yhat"].values
lower = res["yhat_lower"]
upper = res["yhat_upper"]
anomaly = res["anomalies"]

report = get_report(res)
report.to_excel(f"{bs.data_storage}/report-{atm_id}-prophet.xlsx", index=False)

t = res["ds"].values
plt.plot(t, proph)
plt.fill_between(t, lower, upper, color="orange", alpha=0.5)
plt.plot(t, anomaly, "o", color="red", linewidth=2)

plt.legend(["real", f"prophet", "Abnormal", "bounds"])
plt.title(f"ATM ID = {atm_id}, R2 = {score['R2']:0.3f}, MSE: {score['MSE']:0.3f}, MAE: {score['MAE']:0.3f})")

model2.model.plot_components(res)

plt.show()
plt.close()

# AR prediction
# [24, 48, 72, 96, 120, 144, 168, 336, 504]

lag = (np.array([24, 48, 72, 96, 120, 144, 168, 336, 504]) / np.round(pd.Timedelta(freq).total_seconds() / 3600)).astype(int)

ar_model = ARModel(lag)
ar_model.fit(transactions_hourly)

res = ar_model.predict(transactions_hourly)
score = ar_model.score(transactions_hourly)

res = map_anomalies(res)
ar = res["yhat"].values
lower = res["yhat_lower"]
upper = res["yhat_upper"]
anomaly = res["anomalies"]

report = get_report(res)
report.to_excel(f"{bs.data_storage}/report-{atm_id}-AR.xlsx", index=False)

plt.plot(t, data)
plt.plot(t, ar)
plt.fill_between(t, lower, upper, color="orange", alpha=0.5)
plt.plot(t, anomaly, "o", color="red")

plt.legend(["real", f"AR", "Abnormal", "bounds"])
plt.title(f"ATM ID = {atm_id}, R2 = {score['R2']:0.3f}, MSE: {score['MSE']:0.3f}, MAE: {score['MAE']:0.3f})")

plt.show()
