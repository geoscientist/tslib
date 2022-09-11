#encoding utf-8#

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext
import pandas as pd
from datetime import date, timedelta

SPARK_MASTER = 'spark://dhadoop-m01.dvl.mc:7077'
HIVE_METASTORE_WAREHOUSE_DIR = 'hdfs://dhadoop-m01.dvl.mc:9000/user/hive/warehouse'
METASTORE_WAREHOUSE_DIR = 'hdfs://dhadoop-m01.dvl.mc:9000/user/hive/warehouse'
HIVE_METASTORE_URIS = 'thrift://dhadoop-m01.dvl.mc:9083'
SPARK_EXECUTOR_MEMORY = '6G'
SPARK_CORES_MAX = '12'


def get_date_range(start, end):
    sdate = date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))   # start date
    edate = date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))   # end date

    delta = edate - sdate

    drange = []       

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        drange.append(day)

    return drange

def get_transactions_data(start_date, end_date):
    start_date, end_date = str(start_date), str(end_date)
    data = pd.DataFrame()

    spark = SparkSession.builder \
        .master(SPARK_MASTER) \
        .appName('transactions_update') \
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
        .config("spark.cores.max", SPARK_CORES_MAX) \
        .config('hive.metastore.uris', HIVE_METASTORE_URIS) \
        .config('spark.sql.orc.impl', 'hive') \
        .enableHiveSupport() \
        .getOrCreate()

    drng = get_date_range(start_date, end_date)

    for date in drng:
        
        print(f"Downloading date: {date}")

        query = "SELECT ter.name as name, from_unixtime(unix_timestamp(CAST(time_ as timestamp),'yyyy-MM-dd hh:mm:ss'),'yyyy-MM-dd HH:00:00') as date, "
        query += " COUNT(1) as tr_count, "
        query += " COUNT(DISTINCT tran.trancode) as tr_types_num, COLLECT_LIST(tran.trancode) as codelist "
        query += " FROM payments.transacts tran JOIN payments.ter_common ter "
        query += " ON ter.id=tran.origid AND ter.host=tran.host "
        query += " WHERE tran.host='SH' AND tran.termclass=1 AND FLOOR(tran.posentrymode / 10) = 7"
        query += f" AND time_year={date.year} AND time_month={date.month:02} AND time_day={date.day:02} "
        query += " GROUP BY ter.name, date; "

        data_day = spark.sql(query).toPandas()
        data = data.append(data_day)

    return data

df = get_transactions_data("2020-01-01", "2021-08-28")
df.to_csv("transactions_nfc.csv")
