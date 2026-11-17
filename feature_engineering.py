
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, min, max,
    countDistinct, from_unixtime, year, month
)

# =====================================
# 1. CREATE MEMORY-OPTIMIZED SPARK
# =====================================

spark = SparkSession.builder \
    .appName("MovieLens_Feature_Engineering") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "4") \
    .config("spark.driver.maxResultSize", "512m") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\nSpark Session Started Successfully!")


# =====================================
# 2. LOAD ONLY REQUIRED DATA
# =====================================

train = spark.read.parquet(
    "data/processed/train"
).select(
    "userId", "movieId", "rating", "timestamp"
)

movies = spark.read.parquet(
    "data/processed/movies"
).select(
    "movieId", "title", "genres"
)

print("\nTraining data loaded.")
print("Movie data loaded.")


# =====================================
# 3. USER FEATURES
# =====================================

print("\nCreating user features...")

user_features = train.groupBy("userId").agg(
    count("*").alias("total_ratings"),
    avg("rating").alias("average_rating"),
    min("timestamp").alias("first_rating_timestamp"),
    max("timestamp").alias("last_rating_timestamp")
)

user_features.write.mode("overwrite").parquet(
    "data/processed/user_features"
)

print("User features saved.")


# =====================================
# 4. MOVIE FEATURES
# =====================================

print("\nCreating movie features...")

movie_features = train.groupBy("movieId").agg(
    count("*").alias("total_ratings"),
    avg("rating").alias("average_rating"),
    countDistinct("userId").alias("unique_users")
)

movie_features = movie_features.join(
    movies,
    on="movieId",
    how="left"
)

movie_features.write.mode("overwrite").parquet(
    "data/processed/movie_features"
)

print("Movie features saved.")


# =====================================
# 5. INTERACTION FEATURES
# =====================================

print("\nCreating interaction features...")

interactions = train.withColumn(
    "rating_datetime",
    from_unixtime(col("timestamp")).cast("timestamp")
)

interactions = interactions.withColumn(
    "rating_year",
    year("rating_datetime")
)

interactions = interactions.withColumn(
    "rating_month",
    month("rating_datetime")
)

interactions = interactions.join(
    movies,
    on="movieId",
    how="left"
)

interactions.write.mode("overwrite").parquet(
    "data/processed/interactions"
)

print("Interaction features saved.")


# =====================================
# 6. STOP SPARK
# =====================================

spark.stop()

print("\nFeature Engineering Completed Successfully!")
