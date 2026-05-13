"""Tests for dataset loaders using minimal in-memory files."""

import sys
import os
import csv
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from types import SimpleNamespace
from llm_recsys.data.datasets.movielens import MovieLensDataset


# ── MovieLens ─────────────────────────────────────────────────────────────────

def _write_ml_csv(tmp_dir: Path) -> None:
    dataset_dir = tmp_dir / "movielens_1m"
    dataset_dir.mkdir(parents=True)

    with open(dataset_dir / "movies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        w.writerows([
            [1, "Toy Story (1995)", "Animation|Children's|Comedy"],
            [2, "Jumanji (1995)", "Adventure|Children's|Fantasy"],
            [3, "Grumpier Old Men (1995)", "Comedy|Romance"],
        ])

    with open(dataset_dir / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        # user 1: 5 interactions
        for i, (mid, rating, ts) in enumerate([
            (1, 5.0, 100), (2, 4.0, 200), (3, 3.0, 300),
            (1, 4.0, 400), (2, 5.0, 500),
        ]):
            w.writerow([1, mid, rating, ts])
        # user 2: only 3 interactions
        for mid, rating, ts in [(1, 3.0, 100), (3, 4.0, 200), (2, 5.0, 300)]:
            w.writerow([2, mid, rating, ts])
        # user 3: too short (2 items < min_history_len=3)
        w.writerow([3, 1, 5.0, 100])
        w.writerow([3, 2, 4.0, 200])


def make_ml_cfg(tmp_dir: Path, min_history_len: int = 3, max_history_len: int = 20,
                rating_threshold: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        name="movielens_1m",
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        rating_threshold=rating_threshold,
        include_ratings=True,
        include_genres=True,
    )


def test_movielens_loads_items():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        ds = MovieLensDataset(make_ml_cfg(tmp_dir), str(tmp_dir))
        ds.load()
        assert len(ds.item_meta) == 3
        assert "1" in ds.item_meta
        assert ds.item_meta["1"]["title"] == "Toy Story (1995)"


def test_movielens_loads_users():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        ds = MovieLensDataset(make_ml_cfg(tmp_dir), str(tmp_dir))
        ds.load()
        # user 3 has only 2 items and should be excluded
        assert "3" not in ds.users
        assert "1" in ds.users
        assert "2" in ds.users


def test_movielens_history_sorted_by_time():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        ds = MovieLensDataset(make_ml_cfg(tmp_dir), str(tmp_dir))
        ds.load()
        ts = ds.users["1"].timestamps
        assert ts == sorted(ts)


def test_movielens_max_history_len_truncated():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        cfg = make_ml_cfg(tmp_dir, max_history_len=3)
        ds = MovieLensDataset(cfg, str(tmp_dir))
        ds.load()
        for h in ds.users.values():
            assert len(h.item_ids) <= 3


def test_movielens_rating_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        cfg = make_ml_cfg(tmp_dir, rating_threshold=4.0, min_history_len=1)
        ds = MovieLensDataset(cfg, str(tmp_dir))
        ds.load()
        for h in ds.users.values():
            assert h.ratings is not None
            for r in h.ratings:
                assert r >= 4.0


def test_movielens_all_item_ids():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        ds = MovieLensDataset(make_ml_cfg(tmp_dir), str(tmp_dir))
        ds.load()
        assert set(ds.all_item_ids) == {"1", "2", "3"}


def test_movielens_ratings_stored():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_ml_csv(tmp_dir)
        ds = MovieLensDataset(make_ml_cfg(tmp_dir), str(tmp_dir))
        ds.load()
        h = ds.users["2"]
        assert h.ratings is not None
        assert len(h.ratings) == len(h.item_ids)
