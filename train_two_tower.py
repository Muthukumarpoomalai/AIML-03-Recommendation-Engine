
import os
import gc

import numpy as np
import pyarrow.dataset as ds
import tensorflow as tf

from two_tower_model import (
    load_vocabulary,
    build_two_tower_model
)


# =====================================
# CONFIGURATION
# =====================================

TRAIN_PATH = "data/tfrecommender/positive_interactions"

MODEL_DIR = "models/two_tower"

BATCH_SIZE = 512

SCANNER_BATCH_SIZE = 4096

EPOCHS = 1

RANDOM_SEED = 42

print("TensorFlow:", tf.__version__)


# =====================================
# LOAD VOCABULARIES
# =====================================

print("\nLoading vocabularies...")

user_vocab = load_vocabulary(
    "data/processed/user_vocabulary",
    "user_id"
)

movie_vocab = load_vocabulary(
    "data/processed/movie_vocabulary",
    "movie_id"
)

print("Users:", len(user_vocab))
print("Movies:", len(movie_vocab))


# =====================================
# BUILD TWO-TOWER MODEL
# =====================================

print("\nBuilding Two-Tower model...")

user_model, movie_model, model = (
    build_two_tower_model(
        user_vocab,
        movie_vocab
    )
)

print("User tower created.")
print("Movie tower created.")


# =====================================
# LOAD FULL POSITIVE DATASET
# =====================================

print("\nLoading positive interaction dataset...")

if not os.path.exists(TRAIN_PATH):

    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_PATH}"
    )

dataset = ds.dataset(
    TRAIN_PATH,
    format="parquet"
)

total_records = dataset.count_rows()

print("Total training interactions:", total_records)

if total_records == 0:

    raise ValueError("Training dataset is empty")


# =====================================
# GENERATE SHUFFLED BATCHES
# =====================================

def generate_batches():

    rng = np.random.default_rng(RANDOM_SEED)

    scanner = dataset.scanner(
        columns=["userId", "movieId"],
        batch_size=SCANNER_BATCH_SIZE,
        use_threads=True
    )

    for record_batch in scanner.to_batches():

        users = np.asarray(
            record_batch.column("userId").to_pylist(),
            dtype=str
        )

        movies = np.asarray(
            record_batch.column("movieId").to_pylist(),
            dtype=str
        )

        # Shuffle interactions within each
        # Arrow batch to improve in-batch negatives.
        permutation = rng.permutation(len(users))

        users = users[permutation]
        movies = movies[permutation]

        # Split into training batches
        for start in range(0, len(users), BATCH_SIZE):

            end = start + BATCH_SIZE

            user_batch = users[start:end]
            movie_batch = movies[start:end]

            yield {
                "user_id": np.asarray(
                    [x.encode("utf-8") for x in user_batch],
                    dtype=object
                ),

                "movie_id": np.asarray(
                    [x.encode("utf-8") for x in movie_batch],
                    dtype=object
                )
            }


# =====================================
# CREATE TENSORFLOW DATASET
# =====================================

output_signature = {

    "user_id": tf.TensorSpec(
        shape=(None,),
        dtype=tf.string
    ),

    "movie_id": tf.TensorSpec(
        shape=(None,),
        dtype=tf.string
    )
}

train_dataset = tf.data.Dataset.from_generator(
    generate_batches,
    output_signature=output_signature
)

# Shuffle batches without loading the
# entire dataset into RAM.
train_dataset = train_dataset.shuffle(
    buffer_size=128,
    seed=RANDOM_SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.prefetch(1)


# =====================================
# SAVE MODEL AFTER EACH EPOCH
# =====================================

os.makedirs(MODEL_DIR, exist_ok=True)


class TowerCheckpoint(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        print("\nSaving model checkpoint...")

        self.model.user_model.save(
            os.path.join(
                MODEL_DIR,
                "user_tower.keras"
            )
        )

        self.model.movie_model.save(
            os.path.join(
                MODEL_DIR,
                "movie_tower.keras"
            )
        )

        print("Checkpoint saved.")


# =====================================
# TRAIN MODEL
# =====================================

print("\n" + "=" * 50)
print("STARTING TWO-TOWER TRAINING")
print("=" * 50)

print("Training records:", total_records)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[
        TowerCheckpoint()
    ],
    verbose=1
)


# =====================================
# FINAL SAVE
# =====================================

user_model.save(
    os.path.join(
        MODEL_DIR,
        "user_tower.keras"
    )
)

movie_model.save(
    os.path.join(
        MODEL_DIR,
        "movie_tower.keras"
    )
)


# =====================================
# TRAINING RESULTS
# =====================================

final_loss = history.history["loss"][-1]

print("\n" + "=" * 50)
print("TWO-TOWER TRAINING RESULTS")
print("=" * 50)

print("Training records:", total_records)

print("Final training loss:", round(float(final_loss), 6))

print(
    "User tower:",
    os.path.join(MODEL_DIR, "user_tower.keras")
)

print(
    "Movie tower:",
    os.path.join(MODEL_DIR, "movie_tower.keras")
)

print("\nTWO-TOWER TRAINING COMPLETED!")

gc.collect()
