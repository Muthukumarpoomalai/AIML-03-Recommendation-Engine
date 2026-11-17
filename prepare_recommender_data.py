
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# =====================================
# 1. CREATE SPARK SESSION
# =====================================

spark = (
    SparkSession.builder
    .appName("Prepare_TFRS_Data")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# =====================================
# 2. CREATE OUTPUT DIRECTORY
# =====================================

os.makedirs("data/tfrecommender", exist_ok=True)

# =====================================
# 3. LOAD TRAINING INTERACTIONS
# =====================================

train = spark.read.parquet("data/processed/train")

train = train.select(
    col("userId").cast("string").alias("userId"),
    col("movieId").cast("string").alias("movieId"),
    col("rating")
)

print("Training data loaded:", train.count())

# =====================================
# 4. CREATE POSITIVE INTERACTIONS
# =====================================

positive_interactions = train.filter(
    col("rating") >= 4.0
)

print("Positive interactions:", positive_interactions.count())

# =====================================
# 5. LOAD MOVIE INFORMATION
# =====================================

movies = spark.read.parquet("data/processed/movies")

movies = movies.select(
    col("movieId").cast("string").alias("movieId"),
    col("title"),
    col("genres")
)

print("Movie records:", movies.count())

# =====================================
# 6. SAVE DATASETS
# =====================================

train.write.mode("overwrite").parquet(
    "data/tfrecommender/train_interactions"
)

positive_interactions.write.mode("overwrite").parquet(
    "data/tfrecommender/positive_interactions"
)

movies.write.mode("overwrite").parquet(
    "data/tfrecommender/movies"
)

print("\nRecommendation training datasets saved successfully!")

spark.stop()
