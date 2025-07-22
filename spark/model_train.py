from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sqlalchemy import create_engine


def detect_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def cap_outliers(df, col_name, lower_bound, upper_bound):
    df[col_name] = np.where(df[col_name] < lower_bound, lower_bound, df[col_name])
    df[col_name] = np.where(df[col_name] > upper_bound, upper_bound, df[col_name])
    return df









spark = SparkSession.builder \
    .appName("ChurnModelTraining") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/user/telco/cleaned/telco_cleaned.parquet")

df = df.toPandas()


lower, upper = detect_outliers(df, "totalcharges")
df = cap_outliers(df, "totalcharges", lower, upper)
lower, upper = detect_outliers(df, "monthlycharges")
df = cap_outliers(df, "monthlycharges", lower, upper)
lower, upper = detect_outliers(df, "tenure")
df = cap_outliers(df, "tenure", lower, upper)



string_cols = [col for col in df.columns if df[col].dtype == "object" and col not in ["customerid","gender", "partner", "dependents", "phoneservice", "multiplelines", "phoneservice", "onlinebackup", "deviceprotection", "streamingtv", "streamingmovies"]]
numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ["churn","totalcharges"]]
features = string_cols + numeric_cols



X = df[features]
y = df["churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), string_cols),
    ("sc", StandardScaler(), numeric_cols)
])


model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("GBC", GradientBoostingClassifier(random_state=42))
])


model_pipeline.fit(X_train, y_train)

y_pred =model_pipeline.predict_proba(X_test)[:, 1]

threshold = 0.4

y_pred_GBC = (y_pred >= threshold).astype(int)


X_test_with_id = X_test.copy()
X_test_with_id["customerid"] = df.loc[X_test.index, "customerid"]
X_test_with_id["churn"] = y_test.values


X_test_with_id["churn_prediction"] = y_pred_GBC


final_df = X_test_with_id[["customerid", "churn", "churn_prediction"]]

print(final_df.head())


spark_df = spark.createDataFrame(final_df)


PSQL_USERNAME = "postgres"
PSQL_SERVERNAME = "host.docker.internal"
PSQL_PORTNUMBER = 5432
PSQL_DBNAME = "churn_analysis"
PSQL_PASSWORD = "011145"

url = f"jdbc:postgresql://{PSQL_SERVERNAME}:{PSQL_PORTNUMBER}/{PSQL_DBNAME}"

spark_df.write\
    .format("jdbc")\
    .option("url", url)\
    .option("dbtable", "model.model_predictions")\
    .option("user", PSQL_USERNAME)\
    .option("password", PSQL_PASSWORD)\
    .option("driver", "org.postgresql.Driver")\
    .mode("overwrite")\
    .save()




