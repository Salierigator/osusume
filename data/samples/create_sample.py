"""Write data/samples/clean/: a few rows of each table in data/clean, as CSV.

Reads data/clean/anime.parquet, anime_relations.parquet, users.parquet and ratings/*.parquet (streamed,
filtered on user_id). Writes anime_sample.csv (5 random anime), users_sample.csv (5 random users),
ratings_sample.csv (every rating of those 5 users), franchises_sample.csv (the anime of the 5 anime's
franchises and of the two franchises on either side of one cut crossover link, with a few columns of the anime
table) and anime_relations_sample.csv (every relation those anime list). Ids are drawn with a fixed seed, so a
rerun writes the same rows. List columns are joined with "; ": an empty list and a null both become an empty cell.

Usage: python data/samples/create_sample.py
"""
import random
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
OUT = Path(__file__).resolve().parent / "clean"
N_ANIME, N_USERS, SEED = 5, 5, 211
MAX_CROSSOVER = 20   # entries of the two franchises around the crossover shown


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


def crossover(anime: pl.DataFrame, relations: pl.DataFrame) -> set:
    """The two franchises around the cut link between the most popular anime, MAX_CROSSOVER entries at most."""
    linkable = relations.filter("franchise_link")["relation"].unique().to_list()
    cuts = relations.filter(~pl.col("franchise_link") & pl.col("relation").is_in(linkable))
    fid = dict(zip(anime["anime_id"], anime["franchise_id"]))
    members = dict(zip(anime["anime_id"], anime["members"]))
    size = dict(anime.group_by("franchise_id").len().iter_rows())
    a, b = max((p for p in cuts.select("anime_id", "related_id").iter_rows()
                if size[fid[p[0]]] + size[fid[p[1]]] <= MAX_CROSSOVER), key=lambda p: members[p[0]] + members[p[1]])
    return {fid[a], fid[b]}


def main() -> None:
    catalog = pl.read_parquet(CLEAN / "anime.parquet")
    relations = pl.read_parquet(CLEAN / "anime_relations.parquet")
    anime = pick(catalog, "anime_id", N_ANIME)
    users = pick(pl.read_parquet(CLEAN / "users.parquet"), "user_id", N_USERS)
    ratings = (pl.scan_parquet(CLEAN / "ratings" / "*.parquet")
                 .filter(pl.col("user_id").is_in(users["user_id"].to_list()))
                 .collect(engine="streaming"))
    shown = set(anime["franchise_id"]) | crossover(catalog, relations)
    franchises = (catalog.filter(pl.col("franchise_id").is_in(shown))
                         .select("franchise_id", "anime_id", "title", "media_type", "episodes", "episode_minutes",
                                 "start_date", "members")
                         .sort("franchise_id", "start_date", "anime_id", nulls_last=True))
    relations = relations.filter(pl.col("anime_id").is_in(franchises["anime_id"].to_list()))
    OUT.mkdir(exist_ok=True)
    for name, df in (("anime", anime), ("users", users), ("ratings", ratings), ("franchises", franchises),
                     ("anime_relations", relations)):
        write_csv(df, name)


if __name__ == "__main__":
    main()
