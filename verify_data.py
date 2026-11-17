
from pathlib import Path
import pyarrow.dataset as ds

base = Path("data/processed")

folders = [
    "train",
    "validation",
    "test",
    "user_features",
    "movie_features",
    "interactions"
]

for name in folders:
    folder = base / name

    if folder.exists():
        dataset = ds.dataset(str(folder), format="parquet")
        count = dataset.count_rows()
        print(f"{name}: {count:,} rows")
    else:
        print(f"{name}: MISSING")
