
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

# =====================================
# 1. LOAD TRAINED MOVIE TOWER
# =====================================

movie_model = tf.keras.models.load_model(
    "models/two_tower/movie_tower.keras"
)

# =====================================
# 2. LOAD TRAINED MOVIE VOCABULARY
# =====================================

vocab_file = next(
    __import__("pathlib").Path(
        "data/processed/movie_vocabulary"
    ).glob("part-*.csv")
)

with open(vocab_file, encoding="utf-8", newline="") as f:
    movie_vocab = [
        row["movie_id"]
        for row in csv.DictReader(f)
    ]

# =====================================
# 3. LOAD AND FILTER MOVIE METADATA
# =====================================

movies = pd.read_parquet(
    "data/tfrecommender/movies"
)

movies["movieId"] = movies["movieId"].astype(str)

movies = (
    movies[movies["movieId"].isin(set(movie_vocab))]
    .drop_duplicates("movieId")
    .set_index("movieId")
    .loc[movie_vocab]
    .reset_index()
)

print("Movies eligible for retrieval:", len(movies))

# =====================================
# 4. GENERATE EMBEDDINGS
# =====================================

movie_embeddings = movie_model.predict(
    np.asarray(movie_vocab, dtype=str),
    batch_size=512,
    verbose=1
).astype("float32")

print("Embedding shape:", movie_embeddings.shape)

# =====================================
# 5. BUILD FAISS INDEX
# =====================================

faiss.normalize_L2(movie_embeddings)

index = faiss.IndexFlatIP(movie_embeddings.shape[1])
index.add(movie_embeddings)

# =====================================
# 6. SAVE INDEX AND METADATA
# =====================================

os.makedirs("models/faiss", exist_ok=True)

faiss.write_index(
    index,
    "models/faiss/movie_index.faiss"
)

movies.to_parquet(
    "models/faiss/movie_metadata.parquet",
    index=False
)

print("Indexed movies:", index.ntotal)
print("FAISS index rebuilt successfully!")
