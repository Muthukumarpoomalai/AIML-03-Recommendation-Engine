
import pyarrow.dataset as ds

paths = {
    "TRAIN": "data/tfrecommender/train_interactions",
    "TEST": "data/processed/test",
}

for name, path in paths.items():
    dataset = ds.dataset(path, format="parquet")
    print(f"\n===== {name} =====")
    print("Schema:", dataset.schema)

    batch = next(dataset.scanner(
        columns=["userId", "movieId"],
        batch_size=5
    ).to_batches())

    print("User samples:", batch.column("userId").to_pylist())
    print("Movie samples:", batch.column("movieId").to_pylist())
    print("User data type:", batch.column("userId").type)
    print("Movie data type:", batch.column("movieId").type)
