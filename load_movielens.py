
from pyspark.sql import SparkSession

# =====================================
# 1. CREATE SPARK SESSION
# =====================================

spark = SparkSession.builder \
    .appName("AIML03_Recommendation_Engine") \
    .master("local[*]") \
    .getOrCreate()

print("\nApache Spark started successfully!")


# =====================================
# 2. DEFINE DATASET PATHS
# =====================================

ratings_path = "data/raw/ml-25m/ratings.csv"
movies_path = "data/raw/ml-25m/movies.csv"


# =====================================
# 3. LOAD MOVIELENS DATASETS
# =====================================

ratings = spark.read.csv(
    ratings_path,
    header=True,
    inferSchema=True
)

movies = spark.read.csv(
    movies_path,
    header=True,
    inferSchema=True
)


# =====================================
# 4. DISPLAY SAMPLE DATA
# =====================================

print("\nRATINGS DATA")

ratings.show(5, truncate=False)

print("\nMOVIES DATA")

movies.show(5, truncate=False)


# =====================================
# 5. DISPLAY SCHEMAS
# =====================================

print("\nRATINGS SCHEMA")

ratings.printSchema()

print("\nMOVIES SCHEMA")

movies.printSchema()


# =====================================
# 6. DATASET COUNTS
# =====================================

total_ratings = ratings.count()

total_movies = movies.count()

unique_users = ratings.select("userId").distinct().count()

print("\nDATASET STATISTICS")

print("Total Ratings:", total_ratings)

print("Total Movies:", total_movies)

print("Unique Users:", unique_users)


# =====================================
# 7. DISPLAY COLUMNS AND DATA TYPES
# =====================================

print("\nRATINGS COLUMNS:", ratings.columns)

print("\nMOVIES COLUMNS:", movies.columns)

print("\nRATINGS DATA TYPES:", ratings.dtypes)

print("\nMOVIES DATA TYPES:", movies.dtypes)


# =====================================
# 8. STOP SPARK SESSION
# =====================================

spark.stop()

print("\nMovieLens dataset loading completed!")
