"""EDA pass: turn the clean tables into small aggregate tables the notebooks plot.

Reads data/clean/anime.parquet and data/clean/ratings/*.parquet (streamed, a few row
groups at a time). Writes
  cleaning/03_eda/tables/   committed: anime_stats.parquet, monthly_activity.csv,
                            status_score.csv, lag_after_start.csv, progress.csv, cooccurrence.npz
  data/eda/                 local: user_stats.parquet (one row per user) and
                            sample_ratings.parquet (every row of 1% of the users)

Usage: python -m cleaning.03_eda.aggregates [--max-groups N]
`--max-groups` reads only the first N row groups and writes nothing.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from cleaning import config, rules

LAG_EDGES = [0, 30, 90, 180, 365, 730, 1825, 3650]
LAG_LABELS = ["before start", "0-30d", "30-90d", "90-180d", "180d-1y", "1-2y", "2-5y", "5-10y", "10y+"]


def user_stats(df: pl.DataFrame, genres: pl.DataFrame) -> pl.DataFrame:
    watched, updated = pl.col("watched"), pl.col("updated_at")
    stats = df.group_by("user_id", maintain_order=True).agg(
        watched.sum().alias("n_watched"),
        (watched & (pl.col("score") >= 7)).sum().alias("n_watched_liked"),
        pl.col("anime_start").dt.year().filter(watched).mean().alias("mean_start_year"),
        pl.col("members").log10().filter(watched).mean().alias("mean_log_members"),
        (pl.col("episodes_watched") / pl.col("episodes")).filter(pl.col("status") == "dropped").mean()
            .alias("dropped_progress"),
        *[(watched & (updated < pl.lit(t))).sum().alias(f"n_watched_before_{t}") for t in config.TIME_SPLITS],
        *[(watched & (updated >= pl.lit(t))).sum().alias(f"n_watched_from_{t}") for t in config.TIME_SPLITS],
    )
    per_day = (df.filter(updated.is_not_null())
                 .group_by("user_id", updated.dt.date().alias("day")).len()
                 .group_by("user_id").agg(pl.len().alias("n_update_days"),
                                          pl.col("len").max().alias("max_rows_per_day")))
    per_genre = (df.filter(watched).join(genres, on="anime_id")
                   .group_by("user_id", "genre").len()
                   .pivot("genre", index="user_id", values="len"))
    return stats.join(per_day, on="user_id", how="left").join(per_genre, on="user_id", how="left")


def anime_stats(df: pl.DataFrame) -> pl.DataFrame:
    status, score = pl.col("status"), pl.col("score")
    return df.group_by("anime_id").agg(
        pl.len().alias("n_ratings"),
        *[(status == s).sum().alias(f"n_{s}") for s in config.STATUSES],
        (score > 0).sum().alias("n_scored"),
        score.cast(pl.Int64).sum().alias("score_sum"),
        (score.cast(pl.Int64) ** 2).sum().alias("score_sq_sum"),
        *[(score == k).sum().alias(f"n_score_{k}") for k in range(1, 11)],
        pl.col("is_rewatching").sum().alias("n_rewatching"),
        (pl.col("updated_at") >= pl.lit(config.TIME_SPLITS[1])).sum().alias(f"n_from_{config.TIME_SPLITS[1]}"),
        pl.col("updated_at").min().alias("first_update"),
    )


def small_tables(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    watched = pl.col("watched")
    lag = (pl.col("updated_at").dt.date() - pl.col("anime_start")).dt.total_days()
    progress = (pl.col("episodes_watched") / pl.col("episodes")).clip(0, 1)
    return {
        "monthly_activity": df.filter(pl.col("updated_at").is_not_null())
                              .group_by(pl.col("updated_at").dt.truncate("1mo").dt.date().alias("month"),
                                        "status").len(),
        "status_score": df.group_by("status", "score").len(),
        "lag_after_start": df.filter(watched & (pl.col("start_precision") == "day") & lag.is_not_null())
                             .group_by(lag.cut(LAG_EDGES, labels=LAG_LABELS, left_closed=True).cast(pl.String)
                                          .alias("lag")).len(),
        "progress": df.filter(pl.col("status").is_in(["dropped", "on_hold", "watching"]) & progress.is_not_null())
                      .group_by("status", (progress * 10).floor().clip(0, 9).cast(pl.Int8).alias("decile")).len(),
    }


def cooccurrence(df: pl.DataFrame, column_of: pl.DataFrame) -> np.ndarray:
    """Counts of users who watched both anime, among the TOP_ANIME most popular ones."""
    hits = df.filter(pl.col("watched")).join(column_of, on="anime_id")
    _, rows = np.unique(hits["user_id"].to_numpy(), return_inverse=True)
    cols = hits["col"].to_numpy()
    x = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(rows.max() + 1 if len(rows) else 0, config.TOP_ANIME))
    return (x.T @ x).toarray().astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-groups", type=int, help="smoke run on the first N row groups, nothing is written")
    args = ap.parse_args()

    anime = pl.read_parquet(config.CLEAN_ANIME)
    item = anime.select("anime_id", "episodes", "members", "start_precision",
                        pl.col("start_date").alias("anime_start"))
    genres = anime.select("anime_id", pl.col("genres").alias("genre")).explode("genre").drop_nulls()
    top = anime.sort("members", descending=True, nulls_last=True).head(config.TOP_ANIME)
    column_of = top.select("anime_id", pl.int_range(pl.len()).alias("col"))

    per_user, per_anime, tables, sample = [], [], {}, []
    cooc = np.zeros((config.TOP_ANIME, config.TOP_ANIME), dtype=np.int64)
    start = time.time()
    paths = sorted(config.CLEAN_RATINGS.glob("*.parquet"))
    for i, batch in enumerate(rules.iter_user_batches(paths, key="user_id", max_groups=args.max_groups)):
        df = (batch.with_columns(pl.col("status").cast(pl.String))
                   .join(item, on="anime_id", how="left")
                   .with_columns(watched=pl.col("status").is_in(config.WATCHED)))
        per_user.append(user_stats(df, genres))
        per_anime.append(anime_stats(df))
        for name, table in small_tables(df).items():
            tables.setdefault(name, []).append(table)
        cooc += cooccurrence(df, column_of)
        sample.append(batch.filter(pl.col("user_id") % config.SAMPLE_MODULO == 0))
        if i % 20 == 0:
            print(f"[+] batch {i}: {time.time() - start:.0f}s", flush=True)

    per_user = pl.concat(per_user, how="diagonal")
    genre_cols = sorted(set(genres["genre"].to_list()) & set(per_user.columns))
    # the pivot emits genres in a different order on every run, so fix it here
    per_user = (per_user.select(pl.exclude(genre_cols), pl.col(genre_cols).fill_null(0))
                        .rename({g: f"genre_{g}" for g in genre_cols}))
    sums = [c for c in per_anime[0].columns if c not in ("anime_id", "first_update")]
    per_anime = (pl.concat(per_anime).group_by("anime_id")
                   .agg(pl.col(sums).sum(), pl.col("first_update").min()).sort("anime_id"))
    tables = {name: pl.concat(parts).group_by(pl.exclude("len")).agg(pl.col("len").sum().alias("n"))
              for name, parts in tables.items()}
    print(f"[i] {per_user.height:,} users, {per_anime.height:,} anime with ratings, {time.time() - start:.0f}s")
    if args.max_groups:
        print(per_user.head(3), per_anime.head(3), {k: v.height for k, v in tables.items()}, cooc[:3, :3])
        return

    config.TABLES.mkdir(parents=True, exist_ok=True)
    config.EDA.mkdir(parents=True, exist_ok=True)
    per_user.write_parquet(config.USER_STATS, compression="zstd")
    pl.concat(sample).write_parquet(config.SAMPLE_RATINGS, compression="zstd")
    per_anime.write_parquet(config.TABLES / "anime_stats.parquet", compression="zstd")
    for name, table in tables.items():
        table.sort(table.columns[:-1]).write_csv(config.TABLES / f"{name}.csv")
    np.savez_compressed(config.TABLES / "cooccurrence.npz", counts=cooc, anime_id=top["anime_id"].to_numpy())
    print(f"[done] tables -> {config.TABLES.relative_to(config.ROOT)}, "
          f"user_stats + sample -> {config.EDA.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
