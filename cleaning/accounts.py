"""Account rules: which accounts are dropped whole, decided from the survey statistics."""
from __future__ import annotations

import polars as pl

from cleaning import config

RULES = ["rule_no_valid_rows", "rule_list_too_large", "rule_watch_time_impossible"]


def flag_accounts(stats: pl.DataFrame) -> pl.DataFrame:
    """Add one boolean column per rule and `dropped` (any rule) to the account statistics."""
    watch_days = pl.col("watch_minutes") / 1440
    return stats.with_columns(
        rule_no_valid_rows=pl.col("n_rows").fill_null(0) == 0,
        rule_list_too_large=(pl.col("n_rows") > config.MAX_LIST_SIZE).fill_null(False),
        rule_watch_time_impossible=(watch_days > config.MAX_WATCH_DAYS).fill_null(False),
    ).with_columns(dropped=pl.any_horizontal(RULES))
