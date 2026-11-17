
import csv
from pathlib import Path

import tensorflow as tf
import tensorflow_recommenders as tfrs


# =====================================
# LOAD COMPLETE VOCABULARY
# =====================================

def load_vocabulary(folder, column):

    folder = Path(folder)
    files = sorted(folder.glob("part-*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No vocabulary CSV found: {folder}"
        )

    vocabulary = []
    seen = set()

    for file_path in files:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                value = str(row[column])

                if value not in seen:
                    vocabulary.append(value)
                    seen.add(value)

    return vocabulary


# =====================================
# BUILD USER TOWER
# =====================================

def build_user_tower(user_vocab):

    return tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(),
            dtype=tf.string
        ),

        tf.keras.layers.StringLookup(
            vocabulary=user_vocab,
            mask_token=None
        ),

        tf.keras.layers.Embedding(
            input_dim=len(user_vocab) + 1,
            output_dim=64
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(64)

    ], name="user_tower")


# =====================================
# BUILD MOVIE TOWER
# =====================================

def build_movie_tower(movie_vocab):

    return tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(),
            dtype=tf.string
        ),

        tf.keras.layers.StringLookup(
            vocabulary=movie_vocab,
            mask_token=None
        ),

        tf.keras.layers.Embedding(
            input_dim=len(movie_vocab) + 1,
            output_dim=64
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(64)

    ], name="movie_tower")


# =====================================
# TWO-TOWER RETRIEVAL MODEL
# =====================================

class TwoTowerModel(tfrs.models.Model):

    def __init__(self, user_model, movie_model):

        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):

        user_embeddings = self.user_model(
            features["user_id"]
        )

        movie_embeddings = self.movie_model(
            features["movie_id"]
        )

        return self.task(
            user_embeddings,
            movie_embeddings
        )


# =====================================
# MODEL BUILDER
# =====================================

def build_two_tower_model(user_vocab, movie_vocab):

    user_model = build_user_tower(user_vocab)

    movie_model = build_movie_tower(movie_vocab)

    retrieval_model = TwoTowerModel(
        user_model,
        movie_model
    )

    retrieval_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=0.05
        )
    )

    return user_model, movie_model, retrieval_model


if __name__ == "__main__":

    user_vocab = load_vocabulary(
        "data/processed/user_vocabulary",
        "user_id"
    )

    movie_vocab = load_vocabulary(
        "data/processed/movie_vocabulary",
        "movie_id"
    )

    user_model, movie_model, retrieval_model = (
        build_two_tower_model(
            user_vocab,
            movie_vocab
        )
    )

    sample_users = tf.constant(user_vocab[:5])

    sample_movies = tf.constant(movie_vocab[:5])

    user_embeddings = user_model(sample_users)

    movie_embeddings = movie_model(sample_movies)

    print("Users:", len(user_vocab))
    print("Movies:", len(movie_vocab))

    print(
        "User embedding shape:",
        user_embeddings.shape
    )

    print(
        "Movie embedding shape:",
        movie_embeddings.shape
    )

    print("\nTWO-TOWER MODEL CREATED SUCCESSFULLY!")
