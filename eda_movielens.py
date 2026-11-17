
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min,
    max,
    to_timestamp,
    from_unixtime
)



spark = SparkSession.builder \
    .appName("MovieLens_EDA") \
    .master("local[*]") \
    .getOrCreate()

print("\nSpark Session Started Successfully!")




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




print("\n--- DATASET COUNTS ---")

print("Total Ratings:", ratings.count())
print("Total Movies:", movies.count())
print("Unique Users:", ratings.select("userId").distinct().count())




print("\n--- MISSING VALUES ---")

for column in ratings.columns:
    missing = ratings.filter(col(column).isNull()).count()
    print(f"{column}: {missing}")

for column in movies.columns:
    missing = movies.filter(col(column).isNull()).count()
    print(f"{column}: {missing}")




print("\n--- DUPLICATE RECORDS ---")

print(
    "Duplicate Ratings:",
    ratings.count() - ratings.dropDuplicates().count()
)

print(
    "Duplicate Movies:",
    movies.count() - movies.dropDuplicates().count()
)




print("\n--- RATING DISTRIBUTION ---")

ratings.groupBy("rating") \
    .agg(count("*").alias("total")) \
    .orderBy("rating") \
    .show()




print("\n--- RATING STATISTICS ---")

ratings.select(
    avg("rating").alias("average_rating"),
    min("rating").alias("minimum_rating"),
    max("rating").alias("maximum_rating")
).show()




print("\n--- TIMESTAMP RANGE ---")

ratings.select(
    from_unixtime(min("timestamp")).alias("earliest_rating"),
    from_unixtime(max("timestamp")).alias("latest_rating")
).show(truncate=False)




print("\n--- RATINGS SAMPLE ---")
ratings.show(5)

print("\n--- MOVIES SAMPLE ---")
movies.show(5)




spark.stop()

print("\nEDA Completed Successfully!")
