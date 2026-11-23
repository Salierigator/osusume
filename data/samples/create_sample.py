"""Write data/samples/clean/: a few random rows of each table in data/clean, as CSV.

Reads data/clean/anime.parquet, users.parquet and ratings/*.parquet (streamed, filtered on
user_id). Writes anime_sample.csv (5 anime), users_sample.csv (5 users) and ratings_sample.csv
(every rating of those 5 users). Ids are drawn with a fixed seed, so a rerun writes the same rows.
List columns are joined with "; ": an empty list and a null both become an empty cell.

Usage: python data/samples/create_sample.py
"""
import random
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
OUT = Path(__file__).resolve().parent / "clean"
N_ANIME, N_USERS, SEED = 5, 5, 211


def pick(df: pl.DataFrame, key: str, n: int) -> pl.DataFrame:
    """n rows drawn at random by `key`, kept in table order."""
    ids = random.Random(SEED).sample(df[key].to_list(), n)
    return df.filter(pl.col(key).is_in(ids))


def write_csv(df: pl.DataFrame, name: str) -> None:
    lists = [c for c, dtype in df.schema.items() if isinstance(dtype, pl.List)]
    path = OUT / f"{name}_sample.csv"
    (df.with_columns(pl.col(lists).list.eval(pl.element().cast(pl.String)).list.join("; "))
       .write_csv(path, datetime_format="%Y-%m-%dT%H:%M:%S%:z"))
    print(f"{path.relative_to(ROOT)}: {df.height:,} rows, {path.stat().st_size:,} B")


def main() -> None:
    anime = pick(pl.read_parquet(CLEAN / "anime.parquet"), "anime_id", N_ANIME)
    users = pick(pl.read_parquet(CLEAN / "users.parquet"), "user_id", N_USERS)
    ratings = (pl.scan_parquet(CLEAN / "ratings" / "*.parquet")
                 .filter(pl.col("user_id").is_in(users["user_id"].to_list()))
                 .collect(engine="streaming"))
    OUT.mkdir(exist_ok=True)
    for name, df in (("anime", anime), ("users", users), ("ratings", ratings)):
        write_csv(df, name)


if __name__ == "__main__":
    main()
