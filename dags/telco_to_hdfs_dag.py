from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator # Note: providers.standard.operators.bash is deprecated, consider using airflow.operators.bash.BashOperator
from datetime import datetime, timedelta
import requests
import os

# Constants
LOCAL_PATH = "/tmp/telco_churn.csv"
HDFS_PATH = "/data/telco/telco_churn.csv"
HDFS_URL = "http://namenode:9870"

# HDFS path to the PostgreSQL JDBC driver JAR
POSTGRES_JDBC_JAR_HDFS_PATH = "hdfs://namenode:8020/user/spark/jars/postgresql-42.7.7.jar"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

def download_csv():
    """Download CSV file from GitHub"""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    response = requests.get(url)
    response.raise_for_status()

    with open(LOCAL_PATH, "wb") as f:
        f.write(response.content)
    print(f"Downloaded file to {LOCAL_PATH}")

def upload_to_hdfs_webhdfs():
    import requests
    import time

    time.sleep(2)

    requests.put(f"{HDFS_URL}/webhdfs/v1/data/telco?op=MKDIRS&user.name=root")

    response = requests.put(
        f"{HDFS_URL}/webhdfs/v1{HDFS_PATH}?op=CREATE&overwrite=true&user.name=root",
        allow_redirects=False
    )

    redirect_url = response.headers['Location']
    with open(LOCAL_PATH, 'rb') as f:
        requests.put(redirect_url, data=f)

    print(f"File uploaded to {HDFS_PATH}")


with DAG("telco_to_hdfs_dag", default_args=default_args, schedule=None, catchup=False) as dag:

    download = PythonOperator(
        task_id="download_csv",
        python_callable=download_csv
    )


    upload = PythonOperator(
        task_id="upload_to_hdfs",
        python_callable=upload_to_hdfs_webhdfs
    )

    run_cleaning = BashOperator(
    task_id="run_spark_cleaning_and_upload",
    bash_command=f"docker exec spark-master /opt/bitnami/spark/bin/spark-submit "
                 f"--master spark://spark-master:7077 "
                 f"/opt/spark-apps/get_clean.py"
)

    model_training_predictions = BashOperator(
        task_id="model_training_predictions",
        bash_command="docker exec spark-master /opt/bitnami/spark/bin/spark-submit /opt/spark-apps/model_train.py"
    )

    download >> upload >> run_cleaning >> model_training_predictions