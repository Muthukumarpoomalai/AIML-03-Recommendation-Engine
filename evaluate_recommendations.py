
import json
import gc
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import tensorflow as tf
import faiss
import tritonclient.http as httpclient

from tritonclient.http import InferInput, InferRequestedOutput

# =========================================
# CONFIGURATION
# =========================================
TEST_PATH = "data/processed/test"
TRAIN_PATH = "data/processed/train"

USER_MODEL_PATH = "models/two_tower/user_tower.keras"
FAISS_PATH = "models/faiss/movie_index.faiss"
METADATA_PATH = "models/faiss/movie_metadata.parquet"

SAMPLE_USERS = 100
CANDIDATE_COUNT = 1000
RERANK_CANDIDATE_COUNT = 100
TOP_K = 10
BATCH_SIZE = 4
RANDOM_SEED = 42

REPORT_PATH = "evaluation_results.json"

# =========================================
# ID FILTER HELPER
# =========================================
def create_user_filter(dataset, user_ids):

    field_type = dataset.schema.field("userId").type

    if str(field_type).startswith("string"):
        values = [str(x) for x in user_ids]
    else:
        values = [int(x) for x in user_ids]

    return ds.field("userId").isin(values)


# =========================================
# METRIC FUNCTIONS
# =========================================
def calculate_recall(actual, recommended):

    if not actual:
        return 0.0

    hits = len(set(actual).intersection(recommended[:TOP_K]))

    return hits / len(actual)


def calculate_ndcg(actual, recommended):

    if not actual:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, movie_id in enumerate(recommended[:TOP_K], 1)
        if movie_id in actual
    )

    ideal_hits = min(len(actual), TOP_K)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


# =========================================
# DCN INFERENCE
# =========================================
def rank_with_dcn(client, user_id, candidate_movie_ids):

    predictions = []

    for start in range(
        0, len(candidate_movie_ids), BATCH_SIZE
    ):

        batch_ids = candidate_movie_ids[
            start:start + BATCH_SIZE
        ]

        count = len(batch_ids)

        user_array = np.full(
            (count, 1), str(user_id), dtype=object
        )

        movie_array = np.array(
            [str(x) for x in batch_ids],
            dtype=object
        ).reshape(-1, 1)

        user_input = InferInput(
            "user_id", [count, 1], "BYTES"
        )

        movie_input = InferInput(
            "movie_id", [count, 1], "BYTES"
        )

        user_input.set_data_from_numpy(user_array)
        movie_input.set_data_from_numpy(movie_array)

        response = client.infer(
            model_name="dcn_ranking",
            inputs=[user_input, movie_input],
            outputs=[InferRequestedOutput("dense_4")]
        )

        predictions.extend(
            response.as_numpy("dense_4").reshape(-1).tolist()
        )

    return np.asarray(predictions)


# =========================================
# LOAD MODELS
# =========================================
print("Loading evaluation components...")

user_model = tf.keras.models.load_model(USER_MODEL_PATH)

index = faiss.read_index(FAISS_PATH)

movies = pd.read_parquet(
    METADATA_PATH
).reset_index(drop=True)

movie_id_col = (
    "movieId" if "movieId" in movies.columns
    else "movie_id"
)

if len(movies) != index.ntotal:
    raise ValueError(
        "FAISS index and movie metadata count mismatch"
    )

movies[movie_id_col] = movies[movie_id_col].astype(int)

movie_ids = movies[movie_id_col].to_numpy()

catalog_ids = set(movie_ids.tolist())

client = httpclient.InferenceServerClient(
    url="localhost:8000"
)

if not client.is_model_ready("dcn_ranking"):
    raise RuntimeError("Triton DCN model is not READY")

print("Models loaded successfully.")

# =========================================
# LOAD TRAINING USERS
# =========================================
print("\nChecking training user coverage...")

train_ds = ds.dataset(
    TRAIN_PATH,
    format="parquet"
)

train_users = set()

for batch in train_ds.scanner(
    columns=["userId"],
    batch_size=100000
).to_batches():

    train_users.update(
        int(x)
        for x in batch.column("userId").to_pylist()
    )

print("Training users:", len(train_users))

# =========================================
# FIND WARM-START TEST USERS
# =========================================
print("\nFinding eligible test users...")

test_ds = ds.dataset(
    TEST_PATH,
    format="parquet"
)

eligible_users = set()

scanner = test_ds.scanner(
    columns=["userId"],
    filter=ds.field("rating") >= 4.0,
    batch_size=65536
)

for batch in scanner.to_batches():

    for user in batch.column("userId").to_pylist():

        user = int(user)

        if user in train_users:
            eligible_users.add(user)

eligible_users = sorted(eligible_users)

if not eligible_users:
    raise RuntimeError(
        "No warm-start test users found"
    )

rng = np.random.default_rng(RANDOM_SEED)

sampled_users = rng.choice(
    eligible_users,
    size=min(SAMPLE_USERS, len(eligible_users)),
    replace=False
).tolist()

print("Eligible users:", len(eligible_users))
print("Sampled users:", len(sampled_users))

# =========================================
# LOAD TRAINING HISTORY
# =========================================
print("\nLoading training history...")

train_filter = create_user_filter(
    train_ds,
    sampled_users
)

train_table = train_ds.to_table(
    columns=["userId", "movieId"],
    filter=train_filter
)

train_df = train_table.to_pandas()

train_df["userId"] = train_df["userId"].astype(int)
train_df["movieId"] = train_df["movieId"].astype(int)

seen_items = (
    train_df.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

del train_df, train_table
gc.collect()

# =========================================
# LOAD TEST INTERACTIONS
# =========================================
print("\nLoading test interactions...")

test_filter = ds.field("userId").isin(
    sampled_users
)

test_table = test_ds.to_table(
    columns=["userId", "movieId", "rating"],
    filter=test_filter
)

test_df = test_table.to_pandas()

test_df["userId"] = test_df["userId"].astype(int)
test_df["movieId"] = test_df["movieId"].astype(int)

test_df = test_df[
    (test_df["rating"] >= 4.0)
    & (test_df["movieId"].isin(catalog_ids))
]

relevant_items = {}

for user_id, group in test_df.groupby("userId"):

    actual = set(group["movieId"].tolist())

    actual -= seen_items.get(user_id, set())

    if actual:
        relevant_items[user_id] = actual

del test_df, test_table
gc.collect()

evaluation_users = [
    user for user in sampled_users
    if user in relevant_items
]

if not evaluation_users:
    raise RuntimeError(
        "No users have eligible unseen relevant test items"
    )

print("Users with eligible test items:", len(evaluation_users))

# =========================================
# EVALUATION
# =========================================
candidate_recall_scores = []

retrieval_recall_scores = []
retrieval_ndcg_scores = []

final_recall_scores = []
final_ndcg_scores = []

print("\nStarting evaluation...")

for position, user_id in enumerate(
    evaluation_users, 1
):

    actual = relevant_items[user_id]

    # -------------------------------------
    # USER EMBEDDING
    # -------------------------------------
    embedding = user_model.predict(
        np.array([str(user_id)], dtype=str),
        verbose=0
    ).astype("float32")

    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(
            f"Unexpected user embedding shape: {embedding.shape}"
        )

    if embedding.shape[1] != index.d:
        raise ValueError(
            "User embedding dimension does not match FAISS index"
        )

    faiss.normalize_L2(embedding)

    # -------------------------------------
    # FAISS TOP 1000 RETRIEVAL
    # -------------------------------------
    _, indices = index.search(
        embedding,
        min(CANDIDATE_COUNT, index.ntotal)
    )

    all_candidate_ids = [
        int(movie_ids[idx])
        for idx in indices[0]
        if idx >= 0
    ]

    # Remove previously watched movies
    previously_seen = seen_items.get(
        user_id, set()
    )

    all_candidate_ids = [
        movie_id
        for movie_id in all_candidate_ids
        if movie_id not in previously_seen
    ]

    if not all_candidate_ids:
        continue

    # -------------------------------------
    # CANDIDATE RECALL @ 1000
    # -------------------------------------
    candidate_hits = len(
        actual.intersection(all_candidate_ids)
    )

    candidate_recall_scores.append(
        candidate_hits / len(actual)
    )

    # -------------------------------------
    # FAISS TOP 10 METRICS
    # -------------------------------------
    retrieval_recall_scores.append(
        calculate_recall(
            actual, all_candidate_ids
        )
    )

    retrieval_ndcg_scores.append(
        calculate_ndcg(
            actual, all_candidate_ids
        )
    )

    # -------------------------------------
    # SELECT TOP 100 FOR DCN
    # -------------------------------------
    candidate_ids = all_candidate_ids[
        :RERANK_CANDIDATE_COUNT
    ]

    # -------------------------------------
    # DCN RERANKING
    # -------------------------------------
    predicted_ratings = rank_with_dcn(
        client,
        user_id,
        candidate_ids
    )

    ranked_ids = [
        movie_id
        for movie_id, _ in sorted(
            zip(candidate_ids, predicted_ratings),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    # -------------------------------------
    # FINAL METRICS
    # -------------------------------------
    final_recall_scores.append(
        calculate_recall(
            actual, ranked_ids
        )
    )

    final_ndcg_scores.append(
        calculate_ndcg(
            actual, ranked_ids
        )
    )

    if position % 10 == 0:
        print(
            f"Processed {position}/{len(evaluation_users)} users"
        )

# =========================================
# FINAL RESULTS
# =========================================
if not final_recall_scores:
    raise RuntimeError(
        "No users could be evaluated"
    )

results = {
    "evaluation_type": "Warm-start",
    "sampled_users": len(sampled_users),
    "users_evaluated": len(final_recall_scores),

    "retrieval_candidates": CANDIDATE_COUNT,
    "reranking_candidates": RERANK_CANDIDATE_COUNT,
    "top_k": TOP_K,

    "candidate_recall_at_1000": round(
        float(np.mean(candidate_recall_scores)), 6
    ),

    "faiss_retrieval_recall_at_10": round(
        float(np.mean(retrieval_recall_scores)), 6
    ),

    "faiss_retrieval_ndcg_at_10": round(
        float(np.mean(retrieval_ndcg_scores)), 6
    ),

    "dcn_final_recall_at_10": round(
        float(np.mean(final_recall_scores)), 6
    ),

    "dcn_final_ndcg_at_10": round(
        float(np.mean(final_ndcg_scores)), 6
    ),

    "relevant_rating_threshold": 4.0,
    "retrieval_model": "Two-Tower + FAISS",
    "ranking_model": "DCN via NVIDIA Triton"
}

with open(REPORT_PATH, "w") as file:
    json.dump(results, file, indent=4)

# =========================================
# DISPLAY RESULTS
# =========================================
print("\n" + "=" * 60)
print("RECOMMENDATION ENGINE EVALUATION")
print("=" * 60)

print("\nUsers evaluated:", results["users_evaluated"])

print("\n--- CANDIDATE GENERATION ---")
print(
    "Candidate Recall@1000:",
    results["candidate_recall_at_1000"]
)

print("\n--- FAISS RETRIEVAL ---")
print(
    "Retrieval Recall@10:",
    results["faiss_retrieval_recall_at_10"]
)

print(
    "Retrieval NDCG@10:",
    results["faiss_retrieval_ndcg_at_10"]
)

print("\n--- DCN FINAL RANKING ---")
print(
    "Final Recall@10:",
    results["dcn_final_recall_at_10"]
)

print(
    "Final NDCG@10:",
    results["dcn_final_ndcg_at_10"]
)

print("\nEvaluation: Chronological test split")
print("Relevant rating threshold: >= 4.0")
print("Retrieval: Two-Tower + FAISS")
print("Ranking: DCN via NVIDIA Triton")

print("\nResults saved to:", REPORT_PATH)
print("\nMODEL EVALUATION COMPLETED!")
