"""Row rules, the streaming reader and the catalog parsers, on tiny hand-made inputs."""
from collections import Counter
from datetime import date

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from cleaning import catalog, rules

ANIME = pl.DataFrame({"anime_id": [1, 2, 3], "episodes": [12, 24, None]},
                     schema={"anime_id": pl.Int32, "episodes": pl.Int32})
RAW_SCHEMA = {"username": pl.String, "anime_id": pl.String, "status": pl.String, "score": pl.Int8,
              "num_episodes_watched": pl.Int32, "is_rewatching": pl.Int8, "updated_at": pl.String,
              "start_date": pl.String, "finish_date": pl.String}


def raw(rows):
    cols = list(RAW_SCHEMA)
    return pl.DataFrame([dict(zip(cols, r)) for r in rows], schema=RAW_SCHEMA)


def test_fix_ratings_drops_and_nulls():
    df = raw([
        ("a", "1", "completed", 8, 12, 0, "2020-01-02T03:04:05+00:00", "2019-12-01", "2020-01-02"),
        ("a", "2", None, 0, 0, 0, "2020-01-02T03:04:05+00:00", None, None),             # no status
        ("a", "999", "completed", 7, 1, 0, "2020-01-02T03:04:05+00:00", None, None),    # unknown anime
        ("b", "3", "watching", 0, 65535, 1, "1970-01-01T00:00:00+00:00", "2015", "9260-01-01"),
    ])
    counts = Counter()
    out = rules.fix_ratings(df, ANIME, counts)

    assert out["anime_id"].to_list() == [1, 3]
    assert counts["dropped_no_status"] == 1 and counts["dropped_unknown_anime"] == 1
    assert counts["rows_in"] == 4 and counts["rows_out"] == 2
    good, junk = out.row(0, named=True), out.row(1, named=True)
    assert good["start_date"] == date(2019, 12, 1) and good["episodes_watched"] == 12
    assert good["updated_at"].year == 2020 and good["is_rewatching"] is False
    assert junk["updated_at"] is None and junk["start_date"] is None and junk["finish_date"] is None
    assert junk["episodes_watched"] is None and junk["is_rewatching"] is True
    assert counts["nulled_updated_at"] == 1 and counts["nulled_start_date"] == 1
    assert counts["nulled_finish_date"] == 1 and counts["nulled_episodes_watched"] == 1


def test_fix_ratings_keeps_latest_duplicate():
    df = raw([
        ("a", "1", "completed", 8, 12, 0, "2021-01-01T00:00:00+00:00", None, None),
        ("a", "2", "completed", 9, 24, 0, "2020-02-01T00:00:00+00:00", None, None),
        ("a", "1", "watching", 0, 3, 0, "2020-01-01T00:00:00+00:00", None, None),     # older, listed last
        ("a", "2", "watching", 0, 5, 0, "1970-01-01T00:00:00+00:00", None, None),     # junk time, nulled
    ])
    counts = Counter()
    out = rules.fix_ratings(df, ANIME, counts)
    assert counts["dropped_duplicate"] == 2
    assert sorted(out.select("anime_id", "score").rows()) == [(1, 8), (2, 9)]


def test_iter_user_batches_never_splits_a_user(tmp_path):
    # u20 is longer than a batch (2 row groups of 5 rows) and runs across both files.
    names = [f"u{i:02d}" for i in range(40) for _ in range(25 if i == 20 else i % 7 + 1)]
    x = list(range(len(names)))
    cut = names.index("u20") + 12
    paths = [tmp_path / "part-0.parquet", tmp_path / "part-1.parquet"]
    pq.write_table(pa.table({"username": names[:cut], "x": x[:cut]}), paths[0], row_group_size=5)
    pq.write_table(pa.table({"username": names[cut:], "x": x[cut:]}), paths[1], row_group_size=5)

    batches = list(rules.iter_user_batches(paths, groups_per_batch=2))
    assert len(batches) > 3
    assert pl.concat(batches)["x"].to_list() == x
    seen = set()
    for b in batches:
        users = set(b["username"].to_list())
        assert not users & seen
        seen |= users


def test_catalog_parsers():
    assert catalog.jikan_minutes("24 min per ep") == 24
    assert catalog.jikan_minutes("1 hr 30 min") == 90
    assert catalog.jikan_minutes("30 sec") == 0.5
    assert catalog.jikan_minutes("Unknown") is None
    assert catalog._partial_date("2015") == (date(2015, 1, 1), "year")
    assert catalog._partial_date("2015-03") == (date(2015, 3, 1), "month")
    assert catalog._partial_date("2015-03-27") == (date(2015, 3, 27), "day")
    assert catalog._partial_date(None) == (None, None)
