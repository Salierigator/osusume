"""Write data/samples/: the same few random anime and users as rows of data/raw (raw/) and of data/clean (clean/).

Draws 20 anime and 20 users with a fixed seed and keeps them in draw order, so a rerun writes the same rows; only the
first 5 users come with their ratings.

clean/: anime_sample.csv, users_sample.csv and ratings_sample.csv hold their rows of anime.parquet, users.parquet and
ratings/; franchises_sample.csv has franchise_id, anime_id, title and members of the anime in those anime's franchises
and in the two franchises on either side of one cut crossover link; anime_relations_sample.csv has every relation
those anime list. List columns are joined with "; ": an empty list and a null both become an empty cell.

raw/: the lines of anime_mal.jsonl, anime_jikan.jsonl and anime_relations.jsonl for the same anime, as they are (a
file may hold no line or several for one anime); as CSV, the rows of users.csv and ratings/*.parquet for the same
users.

Usage: python data/samples/create_sample.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"
OUT = Path(__file__).resolve().parent
N_ANIME, N_USERS, N_USERS_WITH_RATINGS, SEED = 20, 20, 5, 211
MAX_CROSSOVER = 20   # entries of the two franchises around the crossover shown


def draw(population: int, n: int) -> list:
    """n row positions drawn at random, in draw order."""
    return random.Random(SEED).sample(range(population), n)


def ordered(df: pl.DataFrame, key: str, keys: list) -> pl.DataFrame:
    """The rows of df in the order of keys; rows with the same key keep their order."""
    return df.sort(pl.col(key).replace_strict(keys, list(range(len(keys)))), maintain_order=True)


def report(path: Path, rows: int) -> None:
    print(f"{path.relative_to(ROOT)}: {rows:,} rows, {path.stat().st_size:,} B")


def write_csv(df: pl.DataFrame, folder: str, name: str) -> None:
    lists = [c for c, dtype in df.schema.items() if isinstance(dtype, pl.List)]
    path = OUT / folder / f"{name}_sample.csv"
    (df.with_columns(pl.col(lists).list.eval(pl.element().cast(pl.String)).list.join("; "))
       .write_csv(path, datetime_format="%Y-%m-%dT%H:%M:%S%:z"))
    report(path, df.height)


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


def raw_lines(name: str, key: str, ids: list) -> list:
    """The lines of data/raw/<name>.jsonl for these anime, in the order of ids."""
    wanted, found = set(ids), defaultdict(list)
    with (RAW / f"{name}.jsonl").open("rb") as src:
        for line in src:
            anime_id = json.loads(line)[key]
            if anime_id in wanted:
                found[anime_id].append(line)
    return [line for i in ids for line in found[i]]


def raw_ratings(names: list) -> pl.DataFrame:
    """Every raw rating of these users, read from the row groups whose username dictionary holds one of them."""
    wanted, groups = pa.array(names), []
    for path in sorted((RAW / "ratings").glob("*.parquet")):
        file, usernames = pq.ParquetFile(path), pq.ParquetFile(path, read_dictionary=["username"])
        for g in range(file.num_row_groups):
            column = usernames.read_row_group(g, columns=["username"]).column(0)
            if any(pc.any(pc.is_in(chunk.dictionary, value_set=wanted)).as_py() for chunk in column.chunks):
                groups.append(pl.from_arrow(file.read_row_group(g)))
    return pl.concat(groups).filter(pl.col("username").is_in(names))


def write_clean(catalog: pl.DataFrame, anime: pl.DataFrame, users: pl.DataFrame) -> None:
    relations = pl.read_parquet(CLEAN / "anime_relations.parquet")
    # Unlike is_in, equalities let the row-group statistics on user_id skip all other row groups.
    ratings = (pl.scan_parquet(CLEAN / "ratings" / "*.parquet")
                 .filter(pl.any_horizontal(pl.col("user_id") == u for u in users["user_id"].head(N_USERS_WITH_RATINGS)))
                 .collect(engine="streaming"))
    shown = set(anime["franchise_id"]) | crossover(catalog, relations)
    franchises = (catalog.filter(pl.col("franchise_id").is_in(shown))
                         .sort("franchise_id", "start_date", "anime_id", nulls_last=True)
                         .select("franchise_id", "anime_id", "title", "members"))
    relations = relations.filter(pl.col("anime_id").is_in(franchises["anime_id"].to_list()))
    (OUT / "clean").mkdir(exist_ok=True)
    for name, df in (("anime", anime), ("users", users), ("ratings", ratings), ("franchises", franchises),
                     ("anime_relations", relations)):
        write_csv(df, "clean", name)


def write_raw(ids: list, names: list) -> None:
    (OUT / "raw").mkdir(exist_ok=True)
    for name, key in (("anime_mal", "id"), ("anime_jikan", "mal_id"), ("anime_relations", "id")):
        lines = raw_lines(name, key, ids)
        path = OUT / "raw" / f"{name}_sample.jsonl"
        path.write_bytes(b"".join(lines))
        report(path, len(lines))
    users = pl.scan_csv(RAW / "users.csv", infer_schema_length=0)   # all text: written back as is
    write_csv(ordered(users.filter(pl.col("username").is_in(names)).collect(), "username", names), "raw", "users")
    write_csv(raw_ratings(names[:N_USERS_WITH_RATINGS]), "raw", "ratings")


def main() -> None:
    catalog = pl.read_parquet(CLEAN / "anime.parquet")
    anime = catalog[draw(catalog.height, N_ANIME)]
    user_ids = draw(pq.ParquetFile(CLEAN / "users.parquet").metadata.num_rows, N_USERS)   # user_id = row position
    users = ordered(pl.scan_parquet(CLEAN / "users.parquet").filter(pl.col("user_id").is_in(user_ids)).collect(),
                    "user_id", user_ids)
    write_clean(catalog, anime, users)
    write_raw(anime["anime_id"].to_list(), users["username"].to_list())


if __name__ == "__main__":
    main()
