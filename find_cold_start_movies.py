
import pandas as pd

all_movies = pd.read_parquet(
    "data/tfrecommender/movies"
)

trained_movies = pd.read_parquet(
    "data/processed/movie_features",
    columns=["movieId"]
)

all_movies["movieId"] = all_movies["movieId"].astype(str)
trained_ids = set(trained_movies["movieId"].astype(str))

unseen = all_movies[
    ~all_movies["movieId"].isin(trained_ids)
]

print("Movies in full metadata:", len(all_movies))
print("Movies without training interactions:", len(unseen))

print("\nSample unseen movies:")

print(
    unseen[["movieId", "title", "genres"]]
    .head(10)
    .to_string(index=False)
)
