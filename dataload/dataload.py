import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext

from dl_config import (
    SPARK_MASTER,
    HIVE_METASTORE_WAREHOUSE_DIR,
    METASTORE_WAREHOUSE_DIR,
    HIVE_METASTORE_URIS,
    SPARK_EXECUTOR_MEMORY,
    SPARK_CORES_MAX
)

from utils import get_date_range

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
        query += " WHERE tran.host='SH' AND tran.termclass=1 "
        query += f" AND time_year={date.year} AND time_month={date.month:02} AND time_day={date.day:02} "
        query += " GROUP BY ter.name, date; "

        data_day = spark.sql(query).toPandas()
        data = data.append(data_day)

    return data

