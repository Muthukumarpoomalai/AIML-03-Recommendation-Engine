
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("Prepare_Recommendation_Vocabulary")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read processed feature tables
users = spark.read.parquet("data/processed/user_features")
movies = spark.read.parquet("data/processed/movie_features")

# Extract unique IDs and sort them
user_ids = (
    users.select(col("userId").cast("string").alias("user_id"))
    .distinct()
    .orderBy("user_id")
)

movie_ids = (
    movies.select(col("movieId").cast("string").alias("movie_id"))
    .distinct()
    .orderBy("movie_id")
)

# Save vocabularies
user_ids.coalesce(1).write.mode("overwrite").csv(
    "data/processed/user_vocabulary",
    header=True
)

movie_ids.coalesce(1).write.mode("overwrite").csv(
    "data/processed/movie_vocabulary",
    header=True
)

print("User vocabulary count:", user_ids.count())
print("Movie vocabulary count:", movie_ids.count())

print("Vocabularies saved successfully!")

spark.stop()
