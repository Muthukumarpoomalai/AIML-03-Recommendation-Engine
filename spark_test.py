
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("AIML03_Recommendation_Engine") \
    .master("local[*]") \
    .getOrCreate()

print("Apache Spark started successfully!")

# Sample movie ratings
data = [
    (101, 1, 5.0),
    (101, 2, 4.0),
    (102, 1, 3.0),
    (103, 3, 4.5)
]

columns = ["userId", "movieId", "rating"]

# Create Spark DataFrame
ratings = spark.createDataFrame(data, columns)

ratings.show()
ratings.printSchema()

spark.stop()
