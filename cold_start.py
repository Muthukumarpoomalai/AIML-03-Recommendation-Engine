
import pandas as pd

# =====================================
# LOAD DATA
# =====================================

movie_features = pd.read_parquet(
    "data/processed/movie_features"
)

all_movies = pd.read_parquet(
    "data/tfrecommender/movies"
)

user_features = pd.read_parquet(
    "data/processed/user_features"
)

movie_features["movieId"] = movie_features["movieId"].astype(str)
all_movies["movieId"] = all_movies["movieId"].astype(str)
user_features["userId"] = user_features["userId"].astype(str)

known_users = set(user_features["userId"])
trained_movies = set(movie_features["movieId"])


# =====================================
# CHECK EXISTING USER
# =====================================

def is_known_user(user_id):
    return str(user_id) in known_users


# =====================================
# POPULARITY-BASED RECOMMENDATIONS
# =====================================

def recommend_popular_movies(top_n=10):

    popular = movie_features.sort_values(
        by="total_ratings",
        ascending=False
    ).head(top_n)

    recommendations = []

    for _, movie in popular.iterrows():

        recommendations.append({
            "movie_id": str(movie["movieId"]),
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "predicted_rating": None,
            "retrieval_score": None
        })

    return recommendations


# =====================================
# CONTENT-BASED COLD START
# =====================================

def recommend_similar_movies(new_movie_id, top_n=10):

    new_movie_id = str(new_movie_id)

    new_movie = all_movies[
        all_movies["movieId"] == new_movie_id
    ]

    if new_movie.empty:
        raise ValueError("Movie ID not found in metadata.")

    if new_movie_id in trained_movies:
        raise ValueError(
            "This movie already exists in the trained catalog."
        )

    movie = new_movie.iloc[0]

    if pd.isna(movie["genres"]):
        return []

    target_genres = set(str(movie["genres"]).split("|"))

    candidates = movie_features[
        movie_features["genres"].notna()
        & (movie_features["movieId"] != new_movie_id)
    ].copy()

    def genre_similarity(genres):

        candidate_genres = set(str(genres).split("|"))
        union = target_genres | candidate_genres

        if not union:
            return 0.0

        return len(target_genres & candidate_genres) / len(union)

    candidates["genre_similarity"] = candidates[
        "genres"
    ].apply(genre_similarity)

    candidates = candidates[
        candidates["genre_similarity"] > 0
    ]

    candidates = candidates.sort_values(
        by=["genre_similarity", "total_ratings"],
        ascending=[False, False]
    ).head(top_n)

    recommendations = []

    for _, candidate in candidates.iterrows():

        recommendations.append({
            "movie_id": str(candidate["movieId"]),
            "title": str(candidate["title"]),
            "genres": str(candidate["genres"]),
            "predicted_rating": None,
            "retrieval_score": float(
                candidate["genre_similarity"]
            )
        })

    return recommendations
