"""Check the invariants of data/clean.

Reads data/clean/{anime,users}.parquet, the footers of data/clean/ratings/*.parquet and one
streamed pass over three of its columns. Writes nothing. Exit code 1 if a check fails.

Usage: python -m cleaning.02_clean.verify
"""
from __future__ import annotations

import json
import sys

import polars as pl
import pyarrow.parquet as pq

from cleaning import config, rules

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def main() -> None:
    anime = pl.read_parquet(config.CLEAN_ANIME)
    known = anime["anime_id"].to_list()
    users = pl.read_parquet(config.CLEAN_USERS)
    report = json.loads(config.CLEAN_REPORT.read_text())
    paths = sorted(config.CLEAN_RATINGS.glob("*.parquet"))

    print("== anime")
    check("anime_id unique and not null", anime["anime_id"].n_unique() == anime.height
          and anime["anime_id"].null_count() == 0, f"{anime.height:,} rows")
    check("title never null", anime["title"].null_count() == 0)

    print("== users")
    n_users = users.height
    check("user_id is 0..n-1 in order", users["user_id"].to_list() == list(range(n_users)), f"{n_users:,} users")
    check("username unique", users["username"].n_unique() == n_users)
    check("every user has ratings", users["n_ratings"].min() >= 1)
    check("cohort always set", users["cohort"].null_count() == 0,
          str(dict(users["cohort"].value_counts().iter_rows())))
    check("favorites only reference known anime",
          users.select(pl.col("favorites").explode().drop_nulls().is_in(known).all()).item())

    print("== ratings: footers")
    rows, previous_max, ordered, bounds = 0, -1, True, {}
    for path in paths:
        meta = pq.ParquetFile(path).metadata
        rows += meta.num_rows
        names = [meta.schema.column(i).name for i in range(meta.num_columns)]
        for g in range(meta.num_row_groups):
            for i, name in enumerate(names):
                st = meta.row_group(g).column(i).statistics
                if st is None or not st.has_min_max:
                    continue
                lo, hi = bounds.get(name, (st.min, st.max))
                bounds[name] = (min(lo, st.min), max(hi, st.max))
            st = meta.row_group(g).column(names.index("user_id")).statistics
            ordered &= st.min >= previous_max
            previous_max = st.max
    expected = report["ratings"]
    check("row count matches the report", rows == expected["rows_written"], f"{rows:,} rows in {len(paths)} files")
    dropped = sum(v for k, v in expected.items() if k.startswith("dropped_"))
    check("rows in = rows written + rows dropped", expected["rows_in"] == rows + dropped,
          f"{expected['rows_in']:,} = {rows:,} + {dropped:,}")
    check("user_id never decreases across row groups", ordered)
    check("user_id covers 0..n-1", bounds["user_id"] == (0, n_users - 1), str(bounds["user_id"]))
    check("score within 0..10", bounds["score"][0] >= 0 and bounds["score"][1] <= 10, str(bounds["score"]))
    check("episodes_watched within the catalog maximum",
          bounds["episodes_watched"][0] >= 0 and bounds["episodes_watched"][1] <= anime["episodes"].max(),
          str(bounds["episodes_watched"]))
    lo, hi = bounds["updated_at"]
    check("updated_at within the valid window", lo >= config.UPDATED_AT_MIN.replace(tzinfo=lo.tzinfo)
          and hi <= config.CRAWL_END.replace(tzinfo=hi.tzinfo), f"{lo} .. {hi}")
    for name in ("start_date", "finish_date"):
        lo, hi = bounds[name]
        check(f"{name} within the valid window", lo >= config.WATCH_DATE_MIN and hi <= config.CRAWL_END.date(),
              f"{lo} .. {hi}")

    print("== ratings: streamed pass over user_id, anime_id, updated_at")
    per_user, unknown, duplicates, unsorted = [], 0, 0, 0
    for batch in rules.iter_user_batches(paths, key="user_id", columns=["user_id", "anime_id", "updated_at"]):
        per_user.append(batch.group_by("user_id", maintain_order=True).agg(pl.len().alias("n")))
        unknown += batch.filter(~pl.col("anime_id").is_in(known)).height
        duplicates += batch.select(pl.struct("user_id", "anime_id").is_duplicated().sum()).item()
        unsorted += batch.select((pl.col("updated_at").diff().over("user_id") < 0).sum()).item()
    per_user = pl.concat(per_user)
    check("each user is one contiguous run", per_user["user_id"].n_unique() == per_user.height)
    check("rows per user match users.n_ratings",
          per_user.sort("user_id")["n"].to_list() == users["n_ratings"].to_list())
    check("every anime_id is in the anime table", unknown == 0, f"{unknown} unknown")
    check("no repeated (user_id, anime_id)", duplicates == 0, f"{duplicates} duplicates")
    check("rows of a user are sorted by updated_at", unsorted == 0, f"{unsorted} out of order")

    print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
