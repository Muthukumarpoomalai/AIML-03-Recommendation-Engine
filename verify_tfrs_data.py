
from pathlib import Path
import pyarrow.dataset as ds

base = Path("data/tfrecommender")

for name in [
    "train_interactions",
    "positive_interactions",
    "movies"
]:
    path = base / name

    if path.exists():
        dataset = ds.dataset(str(path), format="parquet")
        print(f"{name}: {dataset.count_rows():,} rows")
    else:
        print(f"{name}: MISSING")
