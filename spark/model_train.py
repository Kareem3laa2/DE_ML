from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier




# Detect and handle outliers (Winsoization)
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
    .appName("ChurnModelTraining") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/user/telco/cleaned/telco_cleaned.parquet")


## Removing Outliers in TotalCharges (Found In the notebook)
t_lower_bound, t_upper_bound = detect_outliers(df , "totalcharges")

df = cap_outliers(df, "totalcharges", t_lower_bound, t_upper_bound)


# Data Splitting
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)


# Choosing Features based on importance (More Details in the notebook model.ipynb)
trainCols = train_data.dtypes
stringCols = [f for (f , v) in trainCols if v == "string" and f not in  ["customerid" , "streamingmovies", "deviceprotection", "streamingtv", "paymentmethod","contranct","onlinebackup"]]
numericCols = [f for (f, v) in trainCols if ((v == "double") & (f != "churn"))]

strIndexCols = [col+"_index" for col in stringCols]

oheCols = [col+"_ohe" for col in stringCols]

allDataCols = oheCols + numericCols 

# Getting the Pipelinee Ready
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='keep') for col in stringCols]

ohe = OneHotEncoder(inputCols=strIndexCols, outputCols=oheCols)

assembler = VectorAssembler(inputCols=allDataCols, outputCol="features_unscaled")

scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")

rf = RandomForestClassifier(featuresCol="features" , labelCol="churn",predictionCol="churnPrediction")

pipeline = Pipeline(stages = indexers +  [ohe ,  assembler  , scaler , rf])

pipeline_model = pipeline.fit(train_data)

predictions = pipeline_model.transform(test_data)

predictions_final = predictions.select("customerid", "churn", "churnPrediction")


PSQL_USERNAME = "postgres"
PSQL_SERVERNAME = "host.docker.internal"
PSQL_PORTNUMBER = 5432
PSQL_DBNAME = "churn_analysis"
PSQL_PASSWORD = "011145"

url = f"jdbc:postgresql://{PSQL_SERVERNAME}:{PSQL_PORTNUMBER}/{PSQL_DBNAME}"

predictions_final.write\
    .format("jdbc")\
    .option("url", url)\
    .option("dbtable", "model.model_predictions")\
    .option("user", PSQL_USERNAME)\
    .option("password", PSQL_PASSWORD)\
    .option("driver", "org.postgresql.Driver")\
    .mode("overwrite")\
    .save()
