import sys
import argparse
from types import TracebackType
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from tslib.model import ARModel, ModelZoo
from configs import get_config
from tslib.features import add_service_mode_feature, extract_features
from tslib.utils import to_localtime, fix_columns

logging.basicConfig(format = '[%(asctime)s] [LEVEL:%(levelname)s] %(message)s',
                        datefmt = ('%Y-%m-%d %H:%M:%S'),
                        level = logging.INFO,
                        stream = sys.stdout)

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'inference'], default='train')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-o', '--output_file')
    return parser

logging.info("starting application")
parser = createParser()
namespace = parser.parse_args()
CONFIG = get_config(namespace.config)
mode = namespace.mode

lags = (np.array([24, 48, 72, 96, 120, 144, 168, 336, 504]) / np.round(
        pd.Timedelta('1H').total_seconds() / 3600)).astype(int)

nfc_lags = (np.array([24, 48, 72, 96, 120, 144, 168, 336, 504]) / np.round(
        pd.Timedelta('1D').total_seconds() / 3600)).astype(int)

if mode == "train":
    logging.info("train mode started")
    df = pd.read_csv(CONFIG['TRAIN_DATA'])
    model = ARModel(lags, timestamp=CONFIG['TIME_COL'])
else:
    logging.info("inference mode started")
    df = pd.read_csv(namespace.input_file)
    if CONFIG['TYPE'] == 'all':
        model =  ModelZoo().all_model
    elif CONFIG['TYPE'] == 'cash-in':
        model = ModelZoo().cashIn_model
    elif CONFIG['TYPE'] == 'cash-out':
        model = ModelZoo().cashOut_model
    elif CONFIG['TYPE'] == 'nfc':
        model = ModelZoo().nfc_model
    logging.info("model checkpoint successfully loaded")

df = fix_columns(df, CONFIG['TIME_COL'])
df = df[["name", "date", "tr_count"]] # TODO remove this row
atm_skto = pd.read_csv(CONFIG['ATM_REGISTRY_DATA'])
atm_skto = atm_skto[["name", "service_time", "timezone"]]
atm_skto['name'] = atm_skto['name'].astype("string")
df = df.merge(atm_skto, on="name")
df = to_localtime(df, 'date', 'timezone')
df = df[["name", "local_time", "tr_count"]]
df = df.sort_values(["name", "local_time"])
df = df.groupby("name").resample(rule=CONFIG['RESAMPLING_WINDOW'], on="local_time").sum()
df = df.reset_index()
df['tr_count_r'] = df.iloc[::-1].groupby('name').tr_count.rolling(CONFIG['AGG_WINDOW']).sum().reset_index(0,drop=True).iloc[::-1]
df = df.drop(columns=["tr_count"])
df = df.rename(columns={'tr_count_r': 'tr_count'})
#---------------------------------------------------------------------------------------
#df = df.reset_index()
if CONFIG['TYPE'] == 'nfc':
    df = df.groupby('name').filter(lambda x : len(x)>nfc_lags.max())
    start, x, y = extract_features(df, 'local_time', nfc_lags, CONFIG['RESAMPLING_WINDOW'])
else:
    df = df.groupby('name').filter(lambda x : len(x)>lags.max())
    start, x, y = extract_features(df, 'local_time', lags, '4H')
    print(x)

logging.info("feature extraction finished")

if mode == 'train':
    logging.info("start model training")
    model.fit(x.iloc[:, 2:], y)
    model.save("./tslib/checkpoints/tr_nfc_gbm_03.joblib")
    logging.info("model successfully saved")
else:
    predicts = model.predict(x.iloc[:, 2:])
    predicts.set_index('index', inplace=True)
    metadata = x.iloc[:, 0:2]
    print(predicts.shape)
    print(metadata.shape)
    predicts = pd.concat([metadata, predicts], axis=1)
    print(predicts.shape)
    predicts.to_csv('predicts.csv')
