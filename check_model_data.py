
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Check_Two_Tower_Data")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

train = spark.read.parquet("data/processed/train")
users = spark.read.parquet("data/processed/user_features")
movies = spark.read.parquet("data/processed/movie_features")

print("\nTRAINING DATA")
train.printSchema()

print("\nUSER FEATURES")
users.printSchema()

print("\nMOVIE FEATURES")
movies.printSchema()

print("\nSAMPLE TRAINING RECORDS")
train.select("userId", "movieId", "rating").show(5)

spark.stop()

