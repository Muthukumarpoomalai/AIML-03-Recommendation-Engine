
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

# =====================================
# 1. LOAD TRAINED USER TOWER
# =====================================

user_model = tf.keras.models.load_model(
    "models/two_tower/user_tower.keras"
)

# =====================================
# 2. LOAD FAISS INDEX
# =====================================

index = faiss.read_index(
    "models/faiss/movie_index.faiss"
)

# =====================================
# 3. LOAD MOVIE METADATA
# =====================================

movies = pd.read_parquet(
    "models/faiss/movie_metadata.parquet"
)

# =====================================
# 4. SELECT A TEST USER
# =====================================

user_id = "2"

# =====================================
# 5. GENERATE USER EMBEDDING
# =====================================

user_embedding = user_model.predict(
    np.array([user_id], dtype=str),
    verbose=0
).astype("float32")

faiss.normalize_L2(user_embedding)

# =====================================
# 6. SEARCH TOP 10 MOVIES
# =====================================

scores, indices = index.search(user_embedding, 10)

print("\nRecommended Movies for User:", user_id)
print("=" * 60)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]), start=1
):
    movie = movies.iloc[idx]

    print(f"\n{rank}. {movie['title']}")
    print("Genres:", movie["genres"])
    print("Similarity:", round(float(score), 4))

print("\nRecommendation testing completed!")
