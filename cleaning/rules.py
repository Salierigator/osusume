"""Row-level fixes for the raw ratings and the streaming reader both passes share.

`fix_ratings` is the single definition of "a valid rating row": the survey applies it in memory
to count what is left per account, the cleaning step applies it again when it writes the table.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator

import polars as pl
import pyarrow.parquet as pq

from cleaning import config

STATUS = pl.Enum(config.STATUSES)


def iter_user_batches(paths, groups_per_batch: int = config.ROW_GROUPS_PER_BATCH,
                      max_groups: int | None = None, key: str = "username",
                      columns: list[str] | None = None) -> Iterator[pl.DataFrame]:
    """Yield a ratings table in file order, a few row groups at a time, never splitting a user.

    A user's rows are contiguous (raw files: `username`, clean files: `user_id`), so the rows of
    the last user of a batch are held back and prepended to the next one. `max_groups` stops
    early (smoke runs).
    """
    carry = None
    done = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for first in range(0, pf.num_row_groups, groups_per_batch):
            groups = list(range(first, min(first + groups_per_batch, pf.num_row_groups)))
            if max_groups is not None:
                groups = groups[:max(0, max_groups - done)]
            if not groups:
                break
            done += len(groups)
            df = pl.from_arrow(pf.read_row_groups(groups, columns=columns))
            if carry is not None:
                df = pl.concat([carry, df])
            other = (df[key] != df[key][-1]).arg_true()
            if other.len() == 0:
                carry = df
                continue
            cut = other[-1] + 1
            carry = df.slice(cut)
            yield df.slice(0, cut)
    if carry is not None and carry.height:
        yield carry


def fix_ratings(df: pl.DataFrame, anime: pl.DataFrame, counts: Counter) -> pl.DataFrame:
    """Raw rating rows -> typed, valid rows. `anime` needs `anime_id` and `episodes`.

    Drops rows without a status, rows whose anime is not in the catalog and repeated
    (username, anime) pairs (the latest update wins). Nulls out timestamps, watch dates and
    episode counts that cannot be real. `counts` collects how many rows each rule touched.
    """
    max_episodes = anime["episodes"].max()
    counts["rows_in"] += df.height

    before = df.height
    df = df.filter(pl.col("status").is_not_null())
    counts["dropped_no_status"] += before - df.height

    before = df.height
    df = (df.with_columns(pl.col("anime_id").cast(pl.Int32, strict=False))
            .join(anime.select("anime_id"), on="anime_id", how="semi"))
    counts["dropped_unknown_anime"] += before - df.height

    updated = pl.col("updated_at").str.to_datetime("%Y-%m-%dT%H:%M:%S%z", time_unit="ms", strict=False)
    watch_date = lambda name: pl.col(name).str.to_date("%Y-%m-%d", strict=False)   # partial dates -> null
    valid_date = lambda e: pl.when(e.is_between(config.WATCH_DATE_MIN, config.CRAWL_END.date())).then(e)
    df = df.with_columns(
        status=pl.col("status").cast(STATUS),
        score=pl.when(pl.col("score").is_between(0, 10)).then(pl.col("score")).otherwise(0).cast(pl.Int8),
        episodes_watched=pl.when(pl.col("num_episodes_watched").is_between(0, max_episodes))
                           .then(pl.col("num_episodes_watched")).cast(pl.Int32),
        is_rewatching=pl.col("is_rewatching") == 1,
        updated_at=pl.when(updated.is_between(config.UPDATED_AT_MIN, config.CRAWL_END)).then(updated),
        start_date=valid_date(watch_date("start_date")),
        finish_date=valid_date(watch_date("finish_date")),
        _had_updated=pl.col("updated_at").is_not_null(),
        _had_start=pl.col("start_date").is_not_null(),
        _had_finish=pl.col("finish_date").is_not_null(),
    )
    nulled = df.select(
        nulled_updated_at=(pl.col("_had_updated") & pl.col("updated_at").is_null()).sum(),
        nulled_start_date=(pl.col("_had_start") & pl.col("start_date").is_null()).sum(),
        nulled_finish_date=(pl.col("_had_finish") & pl.col("finish_date").is_null()).sum(),
        nulled_episodes_watched=pl.col("episodes_watched").is_null().sum(),
    ).row(0, named=True)
    counts.update(nulled)

    if df.select(pl.struct("username", "anime_id").is_duplicated().any()).item():
        before = df.height
        df = (df.with_columns(_row=pl.int_range(pl.len()))
                .sort("updated_at", "_row", nulls_last=False)
                .unique(subset=["username", "anime_id"], keep="last", maintain_order=True)
                .sort("_row"))
        counts["dropped_duplicate"] += before - df.height
    counts["rows_out"] += df.height
    return df.select("username", "anime_id", "status", "score", "episodes_watched", "is_rewatching",
                     "updated_at", "start_date", "finish_date")
