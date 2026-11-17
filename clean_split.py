
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, min, max

# =====================================
# 1. CREATE SPARK SESSION
# =====================================

spark = SparkSession.builder \
    .appName("MovieLens_Cleaning_and_Splitting") \
    .master("local[*]") \
    .getOrCreate()

print("\nSpark Session Started Successfully!")


# =====================================
# 2. LOAD DATASETS
# =====================================

ratings = spark.read.csv(
    "data/raw/ml-25m/ratings.csv",
    header=True,
    inferSchema=True
)

movies = spark.read.csv(
    "data/raw/ml-25m/movies.csv",
    header=True,
    inferSchema=True
)

print("\nOriginal Ratings:", ratings.count())
print("Original Movies:", movies.count())


# =====================================
# 3. CLEAN RATINGS
# =====================================

ratings_clean = ratings.dropna(
    subset=["userId", "movieId", "rating", "timestamp"]
)

ratings_clean = ratings_clean.dropDuplicates()

ratings_clean = ratings_clean.filter(
    (col("userId") > 0) &
    (col("movieId") > 0) &
    (col("rating") >= 0.5) &
    (col("rating") <= 5.0) &
    (col("timestamp") > 0)
)

print("\nCleaned Ratings:", ratings_clean.count())


# =====================================
# 4. CLEAN MOVIES
# =====================================

movies_clean = movies.dropna(
    subset=["movieId", "title", "genres"]
)

movies_clean = movies_clean.dropDuplicates(["movieId"])

movies_clean = movies_clean.filter(
    col("movieId") > 0
)

print("Cleaned Movies:", movies_clean.count())


# =====================================
# 5. CHRONOLOGICAL SPLITTING
# =====================================

print("\nCalculating chronological split boundaries...")

time_boundaries = ratings_clean.approxQuantile(
    "timestamp",
    [0.80, 0.90],
    0.001
)

train_cutoff = time_boundaries[0]
validation_cutoff = time_boundaries[1]

print("Training cutoff:", train_cutoff)
print("Validation cutoff:", validation_cutoff)


# =====================================
# 6. CREATE TRAIN / VALIDATION / TEST
# =====================================

train = ratings_clean.filter(
    col("timestamp") <= train_cutoff
)

validation = ratings_clean.filter(
    (col("timestamp") > train_cutoff) &
    (col("timestamp") <= validation_cutoff)
)

test = ratings_clean.filter(
    col("timestamp") > validation_cutoff
)


# =====================================
# 7. DISPLAY SPLIT COUNTS
# =====================================

print("\n--- CHRONOLOGICAL SPLIT RESULTS ---")

print("Training records:", train.count())
print("Validation records:", validation.count())
print("Testing records:", test.count())


# =====================================
# 8. CHECK DATE RANGES
# =====================================

print("\n--- TIME RANGE VERIFICATION ---")

for name, dataset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    dataset.select(
        from_unixtime(min("timestamp")).alias("start_date"),
        from_unixtime(max("timestamp")).alias("end_date")
    ).show(truncate=False)


# =====================================
# 9. SAVE PROCESSED DATASETS
# =====================================

train.write.mode("overwrite").parquet(
    "data/processed/train"
)

validation.write.mode("overwrite").parquet(
    "data/processed/validation"
)

test.write.mode("overwrite").parquet(
    "data/processed/test"
)

movies_clean.write.mode("overwrite").parquet(
    "data/processed/movies"
)

print("\nAll cleaned datasets saved successfully!")


# =====================================
# 10. STOP SPARK
# =====================================

spark.stop()

print("\nCleaning and chronological splitting completed!")
