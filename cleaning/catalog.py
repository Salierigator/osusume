"""Build the anime table from the two raw catalogs.

The MAL API file is the primary source (newer, covers every live anime). The Jikan file adds
what MAL does not return (producers, the genre / theme / demographic split) and the anime that
were deleted from MAL after the first crawl but still appear in user lists.
Enum values follow the MAL API spelling (`tv`, `pg_13`, `light_novel`).
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime

import polars as pl

from cleaning import config

SCHEMA = {
    "anime_id": pl.Int32,
    "title": pl.String,
    "title_english": pl.String,
    "title_japanese": pl.String,
    "title_synonyms": pl.List(pl.String),
    "media_type": pl.String,
    "source": pl.String,
    "status": pl.String,
    "rating": pl.String,
    "nsfw": pl.String,
    "episodes": pl.Int32,
    "episode_minutes": pl.Float32,
    "start_date": pl.Date,
    "start_precision": pl.String,
    "end_date": pl.Date,
    "season": pl.String,
    "season_year": pl.Int16,
    "genres": pl.List(pl.String),
    "themes": pl.List(pl.String),
    "demographics": pl.List(pl.String),
    "studios": pl.List(pl.String),
    "producers": pl.List(pl.String),
    "synopsis": pl.String,
    "score": pl.Float32,
    "scored_by": pl.Int32,
    "rank": pl.Int32,
    "popularity": pl.Int32,
    "members": pl.Int32,
    "favorites": pl.Int32,
    "image_url": pl.String,
    "mal_created_at": pl.Datetime("ms", "UTC"),
    "on_mal": pl.Boolean,
}

_DURATION = re.compile(r"(?:(\d+) hr)?\s*(?:(\d+) min)?\s*(?:(\d+) sec)?")


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _snake(value):
    return value.lower().replace(" ", "_").replace("-", "_") if value else None


def _partial_date(text):
    """'2015', '2015-03' or '2015-03-27' -> (date padded to the first day, precision)."""
    if not text:
        return None, None
    parts = [int(p) for p in text.split("-")]
    precision = ("year", "month", "day")[len(parts) - 1]
    return date(*(parts + [1, 1])[:3]), precision


def _jikan_date(aired, side):
    prop = ((aired or {}).get("prop") or {}).get(side) or {}
    if not prop.get("year"):
        return None, None
    precision = "day" if prop.get("day") else "month" if prop.get("month") else "year"
    return date(prop["year"], prop.get("month") or 1, prop.get("day") or 1), precision


def jikan_minutes(text):
    """'24 min per ep', '1 hr 30 min', '45 sec' -> minutes; 'Unknown' -> None."""
    hours, minutes, seconds = _DURATION.match(text or "").groups()
    total = int(hours or 0) * 60 + int(minutes or 0) + int(seconds or 0) / 60
    return total or None


def _names(items):
    return [item["name"] for item in items or []]


def _from_mal(r, kind_of, jikan):
    start, precision = _partial_date(r.get("start_date"))
    end, _ = _partial_date(r.get("end_date"))
    by_kind = {"genres": [], "themes": [], "demographics": []}
    for name in _names(r.get("genres")):
        by_kind[kind_of[name]].append(name)   # KeyError = a tag the Jikan file has never seen
    alt = r.get("alternative_titles") or {}
    season = r.get("start_season") or {}
    seconds = r.get("average_episode_duration")
    return {
        "anime_id": r["id"],
        "title": r["title"],
        "title_english": alt.get("en") or None,
        "title_japanese": alt.get("ja") or None,
        "title_synonyms": alt.get("synonyms") or [],
        "media_type": None if r.get("media_type") == "unknown" else r.get("media_type"),
        "source": r.get("source"),
        "status": r.get("status"),
        "rating": r.get("rating"),
        "nsfw": r.get("nsfw"),
        "episodes": r.get("num_episodes") or None,
        "episode_minutes": seconds / 60 if seconds else None,
        "start_date": start,
        "start_precision": precision,
        "end_date": end,
        "season": season.get("season"),
        "season_year": season.get("year"),
        **by_kind,
        "studios": _names(r.get("studios")),
        "producers": _names(jikan["producers"]) if jikan else None,
        "synopsis": r.get("synopsis") or None,
        "score": r.get("mean"),
        "scored_by": r.get("num_scoring_users") or None,
        "rank": r.get("rank"),
        "popularity": r.get("popularity"),
        "members": r.get("num_list_users"),
        "favorites": r.get("num_favorites"),
        "image_url": (r.get("main_picture") or {}).get("medium"),
        "mal_created_at": datetime.fromisoformat(r["created_at"]),
        "on_mal": True,
    }


def _from_jikan(r):
    start, precision = _jikan_date(r.get("aired"), "from")
    end, _ = _jikan_date(r.get("aired"), "to")
    rating = r.get("rating")
    return {
        "anime_id": r["mal_id"],
        "title": r["title"],
        "title_english": r.get("title_english"),
        "title_japanese": r.get("title_japanese"),
        "title_synonyms": r.get("title_synonyms") or [],
        "media_type": _snake(r.get("type")),
        "source": None if r.get("source") == "Unknown" else _snake(r.get("source")),
        "status": _snake(r.get("status")),
        "rating": _snake(rating.split(" - ")[0]) if rating else None,
        "nsfw": None,
        "episodes": r.get("episodes") or None,
        "episode_minutes": jikan_minutes(r.get("duration")),
        "start_date": start,
        "start_precision": precision,
        "end_date": end,
        "season": r.get("season"),
        "season_year": r.get("year"),
        "genres": _names(r.get("genres")) + _names(r.get("explicit_genres")),
        "themes": _names(r.get("themes")),
        "demographics": _names(r.get("demographics")),
        "studios": _names(r.get("studios")),
        "producers": _names(r.get("producers")),
        "synopsis": r.get("synopsis") or None,
        "score": r.get("score"),
        "scored_by": r.get("scored_by"),
        "rank": r.get("rank"),
        "popularity": r.get("popularity"),
        "members": r.get("members"),
        "favorites": r.get("favorites"),
        "image_url": ((r.get("images") or {}).get("jpg") or {}).get("image_url"),
        "mal_created_at": None,
        "on_mal": False,
    }


def build_anime() -> pl.DataFrame:
    """One row per anime_id, sorted by anime_id."""
    jikan = {r["mal_id"]: r for r in _read_jsonl(config.RAW_ANIME_JIKAN)}   # duplicates: last wins
    kind_of = {}
    for r in jikan.values():
        for kind in ("genres", "explicit_genres", "themes", "demographics"):
            for name in _names(r.get(kind)):
                kind_of[name] = "genres" if kind == "explicit_genres" else kind

    mal = _read_jsonl(config.RAW_ANIME_MAL)
    rows = [_from_mal(r, kind_of, jikan.get(r["id"])) for r in mal]
    on_mal = {r["id"] for r in mal}
    rows += [_from_jikan(r) for mal_id, r in jikan.items() if mal_id not in on_mal]
    return pl.DataFrame(rows, schema=SCHEMA).sort("anime_id")
