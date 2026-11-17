
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import redis
import tritonclient.http as httpclient

from tritonclient.http import InferInput, InferRequestedOutput

# -------------------------------
# CONFIGURATION
# -------------------------------
TRITON_URL = "localhost:8000"
REDIS_HOST = "localhost"
CANDIDATE_COUNT = 100
TOP_K = 10

# -------------------------------
# LOAD MODELS AND INDEX
# -------------------------------
print("Loading recommendation components...")

user_model = tf.keras.models.load_model(
    "models/two_tower/user_tower.keras"
)

index = faiss.read_index(
    "models/faiss/movie_index.faiss"
)

movies = pd.read_parquet(
    "models/faiss/movie_metadata.parquet"
).reset_index(drop=True)

# Identify movie ID column
movie_id_col = next(
    (c for c in ["movieId", "movie_id"] if c in movies.columns),
    None
)

if movie_id_col is None:
    raise ValueError("Movie ID column not found in metadata")

if len(movies) != index.ntotal:
    raise ValueError(
        f"Metadata/index mismatch: {len(movies)} rows, "
        f"{index.ntotal} indexed movies"
    )

# -------------------------------
# CONNECT REDIS AND TRITON
# -------------------------------
cache = redis.Redis(
    host=REDIS_HOST,
    port=6379,
    decode_responses=True
)
cache.ping()

triton = httpclient.InferenceServerClient(
    url=TRITON_URL
)

if not triton.is_model_ready("dcn_ranking"):
    raise RuntimeError("DCN model is not READY in Triton")

print("All components connected successfully!")

# -------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------
def recommend(user_id, top_k=TOP_K):

    user_id = str(user_id)
    cache_key = f"recommendations:user:{user_id}"

    # 1. Redis cache check
    cached = cache.get(cache_key)

    if cached:
        print(f"\nRedis CACHE HIT for user {user_id}")
        return json.loads(cached)

    print(f"\nRedis CACHE MISS for user {user_id}")

    # 2. Generate user embedding
    user_embedding = user_model.predict(
        np.array([user_id], dtype=str),
        verbose=0
    ).astype("float32")

    faiss.normalize_L2(user_embedding)

    # 3. FAISS candidate retrieval
    candidate_scores, candidate_indices = index.search(
        user_embedding,
        min(CANDIDATE_COUNT, index.ntotal)
    )

    valid = candidate_indices[0] >= 0
    indices = candidate_indices[0][valid]
    similarities = candidate_scores[0][valid]

    candidate_movies = movies.iloc[indices].copy()

    movie_ids = candidate_movies[movie_id_col].astype(str).tolist()

    # 4. DCN ranking through Triton
    
    # 4. DCN ranking through Triton (batch size <= 4)
    count = len(movie_ids)
    predicted_ratings = []

    for start in range(0, count, 4):
        end = min(start + 4, count)

        batch_movie_ids = movie_ids[start:end]
        batch_count = len(batch_movie_ids)

        user_array = np.full(
            (batch_count, 1),
            user_id,
            dtype=object
        )

        movie_array = np.array(
            batch_movie_ids,
            dtype=object
        ).reshape(-1, 1)

        user_input = InferInput(
            "user_id", [batch_count, 1], "BYTES"
        )
        movie_input = InferInput(
            "movie_id", [batch_count, 1], "BYTES"
        )

        user_input.set_data_from_numpy(user_array)
        movie_input.set_data_from_numpy(movie_array)

        output = InferRequestedOutput("dense_4")

        response = triton.infer(
            model_name="dcn_ranking",
            inputs=[user_input, movie_input],
            outputs=[output]
        )

        batch_predictions = response.as_numpy(
            "dense_4"
        ).reshape(-1)

        predicted_ratings.extend(
            batch_predictions.tolist()
        )

    predicted_ratings = np.array(predicted_ratings)


    # 5. Sort candidates by predicted rating
    results = []

    for i, (_, movie) in enumerate(candidate_movies.iterrows()):
        results.append({
            "movie_id": str(movie[movie_id_col]),
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "predicted_rating": float(predicted_ratings[i]),
            "retrieval_score": float(similarities[i])
        })

    results.sort(
        key=lambda x: x["predicted_rating"],
        reverse=True
    )

    recommendations = results[:top_k]

    # 6. Store recommendations in Redis
    cache.setex(
        cache_key,
        3600,
        json.dumps(recommendations)
    )

    return recommendations


# -------------------------------
# RUN PIPELINE
# -------------------------------
if __name__ == "__main__":

    test_user = "2"

    recommendations = recommend(test_user)

    print("\n" + "=" * 65)
    print(f"TOP {TOP_K} RECOMMENDATIONS FOR USER {test_user}")
    print("=" * 65)

    for rank, item in enumerate(recommendations, 1):
        print(
            f"\n{rank}. {item['title']}"
            f"\n   Genres: {item['genres']}"
            f"\n   Predicted rating: "
            f"{item['predicted_rating']:.4f}"
        )

    print("\nEND-TO-END RECOMMENDATION PIPELINE COMPLETED!")
