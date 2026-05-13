from pathlib import Path
import pandas as pd
from .base import RecDataset, UserHistory


class MovieLensDataset(RecDataset):
    def load(self) -> None:
        data_dir = Path(self.data_dir) / self.cfg.name
        movies_df = self._load_movies(self._find(data_dir, ["movies.dat", "movies.csv"]))
        self._item_meta = {
            str(row["movieId"]): {"title": row["title"], "genres": row["genres"]}
            for _, row in movies_df.iterrows()
        }

        ratings_df = self._load_ratings(self._find(data_dir, ["ratings.dat", "ratings.csv"]))
        threshold = getattr(self.cfg, "rating_threshold", 0.0)
        if threshold:
            ratings_df = ratings_df[ratings_df["rating"] >= threshold]

        ratings_df = ratings_df.sort_values(["userId", "timestamp"])
        for user_id, group in ratings_df.groupby("userId"):
            item_ids = group["movieId"].astype(str).tolist()
            if len(item_ids) < self.cfg.min_history_len:
                continue
            item_ids = item_ids[-self.cfg.max_history_len:]
            ratings = group["rating"].tolist()[-self.cfg.max_history_len:]
            timestamps = group["timestamp"].tolist()[-self.cfg.max_history_len:]
            self._users[str(user_id)] = UserHistory(
                user_id=str(user_id),
                item_ids=item_ids,
                ratings=ratings if getattr(self.cfg, "include_ratings", False) else None,
                timestamps=timestamps,
                item_metadata=self._item_meta,
            )

    def _find(self, data_dir: Path, candidates: list[str]) -> Path:
        for name in candidates:
            for p in data_dir.rglob(name):
                return p
        raise FileNotFoundError(f"None of {candidates} found under {data_dir}")

    def _load_ratings(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".dat":
            return pd.read_csv(
                path, sep="::", engine="python",
                names=["userId", "movieId", "rating", "timestamp"],
            )
        return pd.read_csv(path)

    def _load_movies(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".dat":
            return pd.read_csv(
                path, sep="::", engine="python",
                names=["movieId", "title", "genres"],
                encoding="latin-1",
            )
        return pd.read_csv(path)
