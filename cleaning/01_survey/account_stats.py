"""Survey pass: one row of statistics per account, from the raw ratings with the row fixes applied.

Reads data/raw/ratings/*.parquet (streamed, a few row groups at a time) and the two anime files.
Writes data/eda/survey/account_stats.parquet and row_fixes.json (how many rows each row rule
touched). Nothing is written under data/clean: this pass only measures, so the account rules in
config.py can be chosen from the numbers.

Usage: python -m cleaning.01_survey.account_stats [--max-groups N]
`--max-groups` reads only the first N row groups and writes *.smoke.* files.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import polars as pl

from cleaning import catalog, config, rules


def account_stats(raw: pl.DataFrame, fixed: pl.DataFrame, anime: pl.DataFrame) -> pl.DataFrame:
    """Per-username statistics of one batch. `raw` gives the row count before the fixes."""
    status, score, updated = pl.col("status"), pl.col("score"), pl.col("updated_at")
    df = fixed.join(anime.select("anime_id", "episode_minutes"), on="anime_id", how="left")
    stats = df.group_by("username", maintain_order=True).agg(
        pl.len().alias("n_rows"),
        *[(status == s).sum().alias(f"n_{s}") for s in config.STATUSES],
        (score > 0).sum().alias("n_scored"),
        *[(score == k).sum().alias(f"n_score_{k}") for k in range(1, 11)],
        ((status == "completed") & (score == 0)).sum().alias("n_completed_unscored"),
        pl.col("is_rewatching").sum().alias("n_rewatching"),
        pl.col("episodes_watched").sum().alias("episodes_watched"),
        (pl.col("episodes_watched") * pl.col("episode_minutes")).sum().alias("watch_minutes"),
        updated.min().alias("first_update"),
        updated.max().alias("last_update"),
        updated.is_null().sum().alias("n_no_timestamp"),
        pl.col("start_date").is_not_null().sum().alias("n_start_date"),
        pl.col("finish_date").is_not_null().sum().alias("n_finish_date"),
    )
    timed = df.filter(updated.is_not_null())
    per_day = (timed.group_by("username", updated.dt.date().alias("day")).len()
                    .group_by("username").agg(pl.len().alias("n_update_days"),
                                              pl.col("len").max().alias("max_rows_per_day")))
    per_second = (timed.group_by("username", "updated_at").len()
                       .group_by("username").agg(pl.col("len").max().alias("max_rows_per_second")))
    n_raw = raw.group_by("username", maintain_order=True).agg(pl.len().alias("n_rows_raw"))
    return (n_raw.join(stats, on="username", how="left")
                 .join(per_day, on="username", how="left")
                 .join(per_second, on="username", how="left"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-groups", type=int, help="smoke run on the first N row groups")
    args = ap.parse_args()

    anime = catalog.build_anime()
    paths = sorted(config.RAW_RATINGS.glob("*.parquet"))
    counts, parts, start = Counter(), [], time.time()
    for i, batch in enumerate(rules.iter_user_batches(paths, max_groups=args.max_groups)):
        fixed = rules.fix_ratings(batch, anime, counts)
        parts.append(account_stats(batch, fixed, anime))
        if i % 20 == 0:
            print(f"[+] batch {i}: {counts['rows_in']:,} rows read, {time.time() - start:.0f}s", flush=True)

    stats = pl.concat(parts)
    if stats["username"].n_unique() != stats.height:
        raise SystemExit("a username appears in two separate runs of rows: the contiguity assumption is broken")

    suffix = ".smoke" if args.max_groups else ""
    out = config.ACCOUNT_STATS.with_name(f"account_stats{suffix}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.write_parquet(out, compression="zstd")
    config.ROW_FIXES.with_name(f"row_fixes{suffix}.json").write_text(json.dumps(dict(counts), indent=2))
    print(f"[done] {stats.height:,} accounts -> {out.relative_to(config.ROOT)} in {time.time() - start:.0f}s")
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
