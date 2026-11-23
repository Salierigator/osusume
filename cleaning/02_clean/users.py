"""Write the clean users table.

Reads data/eda/survey/account_stats.parquet (which accounts have valid ratings, and their
counts), data/raw/users.csv (profiles), data/raw/crawl_state.sqlite (crawl time, cohort) and
data/clean/anime.parquet (favorites may only name anime of the catalog).
Writes data/clean/users.parquet: one row per kept account, `user_id` = position in crawl order,
and the account counts into data/clean/cleaning_report.json.

Usage: python -m cleaning.02_clean.users
"""
from __future__ import annotations

import json
import sqlite3

import polars as pl

from cleaning import accounts, config


def load_profiles() -> pl.DataFrame:
    """users.csv is append-only: the last row of a username is the valid one."""
    raw = (pl.read_csv(config.RAW_USERS, infer_schema_length=0)
             .unique(subset="username", keep="last", maintain_order=True))
    year = pl.col("birthday").str.extract(r"(\d{4})").cast(pl.Int16)
    return raw.select(
        "username",
        gender=pl.col("gender").str.to_lowercase().str.replace("-", "_"),
        birth_year=pl.when(year.is_between(config.BIRTH_YEAR_MIN, config.BIRTH_YEAR_MAX)).then(year),
        location=pl.col("location").str.strip_chars(),
        joined=pl.col("joined").str.to_date("%Y-%m-%d", strict=False),
        favorites=pl.col("favorites_anime").str.split(" ").cast(pl.List(pl.Int32)),
        days_watched=pl.col("days_watched").cast(pl.Float32),
        rewatched=pl.col("rewatched").cast(pl.Int32),
    )


def load_crawl_state() -> pl.DataFrame:
    con = sqlite3.connect(f"file:{config.STATE_DB}?mode=ro&immutable=1", uri=True)
    rows = con.execute("SELECT username, discovered_at, ratings_at, profile_status FROM users "
                       "WHERE ratings_status = 'ok'").fetchall()
    con.close()
    state = pl.DataFrame(rows, schema=["username", "discovered_at", "ratings_at", "profile_status"], orient="row")
    return state.select(
        "username",
        "profile_status",
        cohort=pl.col("discovered_at").replace_strict(config.COHORT_BY_DISCOVERED_AT, default=config.DEFAULT_COHORT),
        ratings_crawled_at=pl.col("ratings_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="ms", time_zone="UTC"),
    )


def main() -> None:
    stats = accounts.flag_accounts(pl.read_parquet(config.ACCOUNT_STATS))
    kept = stats.filter(~pl.col("dropped")).with_columns(user_id=pl.int_range(pl.len(), dtype=pl.Int32))
    scores = sum(k * pl.col(f"n_score_{k}") for k in range(1, 11))
    known = pl.read_parquet(config.CLEAN_ANIME, columns=["anime_id"])["anime_id"].to_list()
    in_catalog = pl.col("favorites").list.eval(pl.element().filter(pl.element().is_in(known)))

    users = (kept.join(load_crawl_state(), on="username", how="left")
                 .join(load_profiles(), on="username", how="left")
                 .with_columns(has_profile=pl.col("joined").is_not_null(),
                               # a failed re-crawl left the old row, whose favorites carry the footer bug
                               stale_favorites=(pl.col("profile_status") == "http_404") & pl.col("favorites").is_not_null())
                 .with_columns(favorites=pl.when(~pl.col("stale_favorites")).then(in_catalog),
                               unknown_favorites=pl.col("favorites").list.len() - in_catalog.list.len())
                 .with_columns(favorites=pl.when(pl.col("favorites").list.len() > 0).then(pl.col("favorites")))
                 .sort("user_id"))
    out = users.select(
        "user_id", "username", "cohort", "ratings_crawled_at", "has_profile",
        "gender", "birth_year", "location", "joined", "favorites", "days_watched", "rewatched",
        pl.col("n_rows").cast(pl.Int32).alias("n_ratings"),
        *[pl.col(f"n_{s}").cast(pl.Int32) for s in config.STATUSES],
        pl.col("n_scored").cast(pl.Int32),
        pl.when(pl.col("n_scored") > 0).then(scores / pl.col("n_scored")).cast(pl.Float32).alias("mean_score"),
        "first_update", "last_update",
    )
    config.CLEAN.mkdir(parents=True, exist_ok=True)
    out.write_parquet(config.CLEAN_USERS, compression="zstd")

    counts = {
        "accounts_in": stats.height,
        **{rule.replace("rule_", "dropped_"): int(stats[rule].sum()) for rule in accounts.RULES},
        "accounts_dropped": int(stats["dropped"].sum()),
        "accounts_kept": out.height,
        "without_profile": int((~out["has_profile"]).sum()),
        "favorites_blanked": int(users["stale_favorites"].sum()),
        "favorite_ids_not_in_catalog": int(users.filter(~pl.col("stale_favorites"))["unknown_favorites"].sum()),
    }
    report = json.loads(config.CLEAN_REPORT.read_text()) if config.CLEAN_REPORT.exists() else {}
    report["users"] = counts
    config.CLEAN_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[done] {out.height:,} users -> {config.CLEAN_USERS.relative_to(config.ROOT)}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
