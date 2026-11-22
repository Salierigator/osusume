"""Write the clean ratings table.

Reads data/raw/ratings/*.parquet (streamed, a few row groups at a time), data/clean/anime.parquet
and data/clean/users.parquet. Writes data/clean/ratings/part-NN.parquet: the valid rows
(cleaning/rules.py) of the accounts kept in the users table, keyed by `user_id`, sorted by
(user_id, updated_at). A user never spans two files. Row counts per rule go to
data/clean/cleaning_report.json.

Usage: python -m cleaning.02_clean.ratings [--max-groups N]
`--max-groups` reads only the first N raw row groups and writes to data/clean/ratings.smoke/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter

import polars as pl
import pyarrow.parquet as pq

from cleaning import config, rules

COLUMNS = ["user_id", "anime_id", "status", "score", "episodes_watched", "is_rewatching",
           "updated_at", "start_date", "finish_date"]
DELTA = ["user_id", "updated_at"]


class PartWriter:
    """Appends tables to part-NN.parquet files, starting a new file once one is full."""

    def __init__(self, folder):
        self.folder, self.writer, self.part, self.rows_in_part = folder, None, -1, 0

    def write(self, df: pl.DataFrame) -> None:
        table = df.to_arrow()
        if self.writer is None or self.rows_in_part >= config.ROWS_PER_PART:
            self.close()
            self.part += 1
            self.rows_in_part = 0
            # Sorted ids and timestamps delta-encode to a quarter less than dictionary pages.
            self.writer = pq.ParquetWriter(
                self.folder / f"part-{self.part:02d}.parquet", table.schema, compression="zstd",
                use_dictionary=[c for c in COLUMNS if c not in DELTA], column_encoding={c: "DELTA_BINARY_PACKED" for c in DELTA})
        self.writer.write_table(table, row_group_size=config.ROW_GROUP_SIZE)
        self.rows_in_part += table.num_rows

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-groups", type=int, help="smoke run on the first N raw row groups")
    args = ap.parse_args()

    anime = pl.read_parquet(config.CLEAN_ANIME, columns=["anime_id", "episodes"])
    users = pl.read_parquet(config.CLEAN_USERS, columns=["user_id", "username"])
    final = config.CLEAN_RATINGS.with_name("ratings.smoke") if args.max_groups else config.CLEAN_RATINGS
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    counts, writer, start = Counter(), PartWriter(tmp), time.time()
    paths = sorted(config.RAW_RATINGS.glob("*.parquet"))
    for i, batch in enumerate(rules.iter_user_batches(paths, max_groups=args.max_groups)):
        fixed = rules.fix_ratings(batch, anime, counts)
        kept = (fixed.join(users, on="username", how="inner")
                     .sort("user_id", "updated_at", "anime_id", nulls_last=False)
                     .select(COLUMNS))
        counts["dropped_account_rows"] += fixed.height - kept.height
        counts["rows_written"] += kept.height
        if kept.height:
            writer.write(kept)
        if i % 20 == 0:
            print(f"[+] batch {i}: {counts['rows_in']:,} rows read, {time.time() - start:.0f}s", flush=True)
    writer.close()

    # The folder is swapped in only after the last row group is on disk.
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    if not args.max_groups:
        report = json.loads(config.CLEAN_REPORT.read_text()) if config.CLEAN_REPORT.exists() else {}
        report["ratings"] = dict(counts)
        config.CLEAN_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[done] {counts['rows_written']:,} rows in {writer.part + 1} files -> "
          f"{final.relative_to(config.ROOT)} in {time.time() - start:.0f}s")
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
