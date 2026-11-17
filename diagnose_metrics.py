
import pyarrow.dataset as ds

TRAIN_PATH = "data/tfrecommender/train_interactions"
TEST_PATH = "data/processed/test"

train = ds.dataset(TRAIN_PATH, format="parquet")
test = ds.dataset(TEST_PATH, format="parquet")

# Collect IDs from training interactions
train_users = set()
train_movies = set()

for batch in train.scanner(
    columns=["userId", "movieId"],
    batch_size=100_000
).to_batches():
    train_users.update(batch.column("userId").to_pylist())
    train_movies.update(batch.column("movieId").to_pylist())

# Collect positive test interactions
test_users = set()
test_movies = set()
positive_interactions = 0

for batch in test.scanner(
    columns=["userId", "movieId", "rating"],
    batch_size=100_000
).to_batches():
    ratings = batch.column("rating").to_pylist()
    users = batch.column("userId").to_pylist()
    movies = batch.column("movieId").to_pylist()

    for user, movie, rating in zip(users, movies, ratings):
        if rating >= 4:
            positive_interactions += 1
            test_users.add(user)
            test_movies.add(movie)

# Normalize IDs to integers before comparison
train_users = {int(x) for x in train_users}
train_movies = {int(x) for x in train_movies}

known_users = test_users & train_users
known_movies = test_movies & train_movies

print("\n========== DATA COVERAGE DIAGNOSTICS ==========")
print("Training users:", len(train_users))
print("Training movies:", len(train_movies))
print("Positive test interactions:", positive_interactions)
print("Positive test users:", len(test_users))
print("Positive test movies:", len(test_movies))

print("\nUser coverage:", round(len(known_users) / max(len(test_users), 1) * 100, 2), "%")
print("Movie coverage:", round(len(known_movies) / max(len(test_movies), 1) * 100, 2), "%")

print("\nUnknown test users:", len(test_users - train_users))
print("Unknown test movies:", len(test_movies - train_movies))
print("===============================================")
