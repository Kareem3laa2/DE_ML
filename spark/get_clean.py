from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def detect_outliers(df, column):
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
    Q1 = quantiles[0]
    Q3 = quantiles[1]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df.filter((col(column) < lower_bound) | (col(column) > upper_bound)) 
    print(f"Number of outliers in {column}: {outliers.count()}")
    return lower_bound, upper_bound

def cap_outliers(df, col_name, lower_bound, upper_bound):
    return df.withColumn(
        col_name,
        when(col(col_name) < lower_bound, lower_bound)
        .when(col(col_name) > upper_bound, upper_bound)
        .otherwise(col(col_name))
    )

spark = SparkSession.builder \
    .appName("TelcoExploration") \
    .master("spark://spark-master:7077") \
    .config("spark.jars", "/opt/bitnami/spark/jars/postgresql-42.7.7.jar") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .getOrCreate()

df = spark.read.csv("hdfs://namenode:8020/data/telco/telco_churn.csv", header=True, inferSchema=True)

df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))

for colname in df.columns:
    df = df.withColumnRenamed(colname, colname.strip().lower().replace(" ", "_"))

df = df.withColumn("totalcharges", when(col("totalcharges").isNull(), 0).otherwise(col("totalcharges")))

df = df.withColumn("churn", when(col("churn") == "Yes", 1).otherwise(0))

df_cleaned = df.withColumn("seniorcitizen", when(col("seniorcitizen") == 1 , "Yes").otherwise("No"))

t_lower_bound, t_upper_bound = detect_outliers(df_cleaned , "totalcharges")

df_cleaned = cap_outliers(df_cleaned, "totalcharges", t_lower_bound, t_upper_bound)



df_cleaned.write.mode("overwrite").parquet("hdfs://namenode:8020/user/telco/cleaned/telco_cleaned.parquet")

PSQL_USERNAME = "postgres"
PSQL_SERVERNAME = "host.docker.internal"
PSQL_PORTNUMBER = 5432
PSQL_DBNAME = "churn_analysis"
PSQL_PASSWORD = "011145"

url = f"jdbc:postgresql://{PSQL_SERVERNAME}:{PSQL_PORTNUMBER}/{PSQL_DBNAME}"

df_cleaned.write\
    .format("jdbc")\
    .option("url", url)\
    .option("dbtable", "churnWH.fact_churn_analysis")\
    .option("user", PSQL_USERNAME)\
    .option("driver", "org.postgresql.Driver")\
    .option("password", PSQL_PASSWORD)\
    .mode("overwrite")\
    .save()

print("The Cleaning is done!")
