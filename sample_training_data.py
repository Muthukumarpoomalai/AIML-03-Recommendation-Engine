
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Create_Two_Tower_Training_Sample")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read positive interactions
positive = spark.read.parquet(
    "data/tfrecommender/positive_interactions"
)

print("Original positive interactions:", positive.count())

# Select a reproducible 5% sample
sample = positive.sample(
    withReplacement=False,
    fraction=0.05,
    seed=42
)

print("Sample training records:", sample.count())

# Save sample
sample.write.mode("overwrite").parquet(
    "data/tfrecommender/training_sample"
)

print("Training sample saved successfully!")

spark.stop()
