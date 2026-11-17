
import os
import numpy as np
import pyarrow.dataset as ds
import tensorflow as tf
from tensorflow.keras import layers, Model

# =====================================
# 1. CONFIGURATION
# =====================================

BATCH_SIZE = 1024
EPOCHS = 3
EMBEDDING_DIM = 32

# =====================================
# 2. LOAD VOCABULARIES
# =====================================

def load_vocab(folder, column):
    import csv
    import glob

    files = glob.glob(os.path.join(folder, "part-*.csv"))

    with open(files[0], encoding="utf-8", newline="") as f:
        return [row[column] for row in csv.DictReader(f)]


user_vocab = load_vocab(
    "data/processed/user_vocabulary", "user_id"
)

movie_vocab = load_vocab(
    "data/processed/movie_vocabulary", "movie_id"
)

print("Users:", len(user_vocab))
print("Movies:", len(movie_vocab))

# =====================================
# 3. CREATE DCN MODEL
# =====================================

user_input = layers.Input(shape=(), dtype=tf.string, name="user_id")
movie_input = layers.Input(shape=(), dtype=tf.string, name="movie_id")

user_lookup = layers.StringLookup(
    vocabulary=user_vocab, mask_token=None
)(user_input)

movie_lookup = layers.StringLookup(
    vocabulary=movie_vocab, mask_token=None
)(movie_input)

user_embedding = layers.Embedding(
    len(user_vocab) + 1, EMBEDDING_DIM
)(user_lookup)

movie_embedding = layers.Embedding(
    len(movie_vocab) + 1, EMBEDDING_DIM
)(movie_lookup)

user_embedding = layers.Flatten()(user_embedding)
movie_embedding = layers.Flatten()(movie_embedding)

features = layers.Concatenate()([
    user_embedding,
    movie_embedding
])

# Cross Network
cross = features

for _ in range(2):
    cross_product = layers.Dense(
        1, use_bias=True
    )(cross)

    cross_product = layers.Multiply()([
        features, cross_product
    ])

    cross = layers.Add()([
        cross_product, cross
    ])

# Deep Network
deep = layers.Dense(128, activation="relu")(features)
deep = layers.Dense(64, activation="relu")(deep)

# Combine Cross + Deep
combined = layers.Concatenate()([cross, deep])

output = layers.Dense(1, activation="linear")(combined)

model = Model(
    inputs=[user_input, movie_input],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =====================================
# 4. STREAM PARQUET DATA
# =====================================

data_path = "data/tfrecommender/ranking_sample"
dataset = ds.dataset(data_path, format="parquet")

print("Ranking records:", dataset.count_rows())


def generate_batches():
    scanner = dataset.scanner(
        columns=["userId", "movieId", "rating"],
        batch_size=BATCH_SIZE
    )

    for batch in scanner.to_batches():
        users = batch.column("userId").to_pylist()
        movies = batch.column("movieId").to_pylist()
        ratings = batch.column("rating").to_numpy().astype("float32")

        yield (
            {
                "user_id": np.asarray(
                    [str(x) for x in users], dtype=str
                ),
                "movie_id": np.asarray(
                    [str(x) for x in movies], dtype=str
                )
            },
            ratings
        )


output_signature = (
    {
        "user_id": tf.TensorSpec((None,), tf.string),
        "movie_id": tf.TensorSpec((None,), tf.string)
    },
    tf.TensorSpec((None,), tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    generate_batches,
    output_signature=output_signature
).prefetch(1)

# =====================================
# 5. TRAIN DCN
# =====================================

print("\nStarting DCN training...")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1
)

# =====================================
# 6. SAVE MODEL
# =====================================

os.makedirs("models/dcn", exist_ok=True)

model.save("models/dcn/dcn_ranking_model.keras")

print("\nFinal training loss:", history.history["loss"][-1])
print("Final training MAE:", history.history["mae"][-1])
print("DCN model saved successfully!")
