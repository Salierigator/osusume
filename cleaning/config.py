"""Paths, row rules and thresholds of the cleaning pipeline.

Every number a rule depends on lives here, so the survey, the cleaning scripts and the EDA
read the same definitions.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
EDA = ROOT / "data" / "eda"

RAW_RATINGS = RAW / "ratings"
RAW_USERS = RAW / "users.csv"
RAW_ANIME_MAL = RAW / "anime_mal.jsonl"
RAW_ANIME_JIKAN = RAW / "anime_jikan.jsonl"
RAW_ANIME_RELATIONS = RAW / "anime_relations.jsonl"
STATE_DB = RAW / "crawl_state.sqlite"

ACCOUNT_STATS = EDA / "survey" / "account_stats.parquet"
ROW_FIXES = EDA / "survey" / "row_fixes.json"
USER_STATS = EDA / "user_stats.parquet"           # per user, local only
SAMPLE_RATINGS = EDA / "sample_ratings.parquet"   # every row of a 1% user sample, local only
TABLES = ROOT / "cleaning" / "03_eda" / "tables"  # small aggregate tables, committed

CLEAN_ANIME = CLEAN / "anime.parquet"
CLEAN_ANIME_RELATIONS = CLEAN / "anime_relations.parquet"
CLEAN_USERS = CLEAN / "users.parquet"
CLEAN_RATINGS = CLEAN / "ratings"
CLEAN_REPORT = CLEAN / "cleaning_report.json"

# --- streaming ---
ROW_GROUPS_PER_BATCH = 16          # ~2M raw rows held in memory at a time
ROWS_PER_PART = 64_000_000         # clean ratings roll over to a new part file after this many rows
ROW_GROUP_SIZE = 1 << 17

# --- row rules (ratings) ---
UPDATED_AT_MIN = datetime(2004, 1, 1, tzinfo=timezone.utc)   # MAL opened in 2004; epoch-0 values are junk
CRAWL_END = datetime(2026, 10, 7, tzinfo=timezone.utc)       # the user crawl closed on 2026-10-06
WATCH_DATE_MIN = date(1960, 1, 1)
STATUSES = ["watching", "completed", "on_hold", "dropped", "plan_to_watch"]

# --- account rules: accounts whose list cannot be one person's viewing history are dropped whole ---
# Chosen from the tails in 01_survey/thresholds.ipynb.
MAX_LIST_SIZE = 10_000     # entries; the whole MAL catalog is ~30.8k anime
MAX_WATCH_DAYS = 1_000     # days of watch time = 6.6 hours every day for ten years

# --- row rules (users) ---
BIRTH_YEAR_MIN, BIRTH_YEAR_MAX = 1940, 2016

# --- anime relations (MAL's types). A franchise follows every type but `character` and `other`:
# those two also join unrelated works (crossovers, collaboration ads, shared music videos). ---
RELATION_TYPES = ["sequel", "prequel", "side_story", "parent_story", "summary", "full_story",
                  "alternative_version", "alternative_setting", "spin_off", "character", "other"]
FRANCHISE_RELATIONS = [t for t in RELATION_TYPES if t not in ("character", "other")]

# --- EDA aggregates ---
WATCHED = ["completed", "watching"]
TIME_SPLITS = [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)]   # candidate train / test cut dates
TOP_ANIME = 200       # co-occurrence is counted among the anime with the most members
SAMPLE_MODULO = 100   # users with user_id % SAMPLE_MODULO == 0 go to the row sample
