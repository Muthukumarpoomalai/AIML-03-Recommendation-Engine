
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Prepare_DCN_Ranking_Data")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read original training interactions
train = spark.read.parquet("data/processed/train")

print("Original training records:", train.count())

# Select required ranking features
ranking_data = train.select(
    "userId",
    "movieId",
    "rating"
)

# Create reproducible 5% sample
ranking_sample = ranking_data.sample(
    withReplacement=False,
    fraction=0.05,
    seed=42
)

print("Ranking sample records:", ranking_sample.count())

# Save sample
ranking_sample.write.mode("overwrite").parquet(
    "data/tfrecommender/ranking_sample"
)

print("DCN ranking data saved successfully!")

spark.stop()
