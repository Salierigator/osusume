"""Relation loading and franchise components, on a tiny hand-made relation file."""
import json

import polars as pl
import pytest

from cleaning import config, relations


def write(tmp_path, rows):
    path = tmp_path / "anime_relations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def rel(aid, kind):
    return {"node": {"id": aid, "title": f"anime {aid}"}, "relation_type": kind}


RAW = [
    {"id": 1, "status": 200, "related_anime": [rel(2, "sequel"), rel(4, "character")]},
    {"id": 2, "status": 200, "related_anime": [rel(1, "prequel"), rel(99, "sequel")]},   # 99: not in the catalog
    {"id": 3, "status": 200, "related_anime": [rel(1, "side_story"), rel(3, "other")]},  # no edge back; self
    {"id": 5, "status": 200, "related_anime": [rel(6, "sequel")]},
    {"id": 6, "status": 404, "related_anime": []},
    {"id": 4, "status": 200, "related_anime": []},
    {"id": 5, "status": 200, "related_anime": [rel(6, "other"), rel(6, "other")]},       # repeated id: last wins
]
CATALOG = [1, 2, 3, 4, 5, 6, 7]


def test_load_keeps_known_distinct_edges(tmp_path):
    df, counts = relations.load_relations(write(tmp_path, RAW), CATALOG)
    assert df.rows() == [(1, 2, "sequel"), (1, 4, "character"), (2, 1, "prequel"),
                         (3, 1, "side_story"), (5, 6, "other")]
    assert df.schema["relation"] == relations.RELATION
    assert counts["lines"] == 6 and counts["status_404"] == 1
    assert (counts["edges_in"], counts["dropped_unknown_anime"], counts["dropped_self"],
            counts["dropped_duplicate"], counts["edges_out"]) == (8, 1, 1, 1, 5)


def test_unknown_relation_type_is_an_error(tmp_path):
    path = write(tmp_path, [{"id": 1, "status": 200, "related_anime": [rel(2, "crossover")]}])
    with pytest.raises(ValueError, match="crossover"):
        relations.load_relations(path, [1, 2])


def test_franchise_follows_strong_relations_both_ways(tmp_path):
    df, _ = relations.load_relations(write(tmp_path, RAW), CATALOG)
    fr = relations.franchise_ids(CATALOG, df, config.FRANCHISE_RELATIONS)
    # 1-2 sequel, 3 -> 1 side story listed on one side only; 1-4 character and 5-6 other do not count
    assert dict(zip(CATALOG, fr.to_list())) == {1: 1, 2: 1, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    assert fr.dtype == pl.Int32


def test_franchise_id_is_the_smallest_id_whatever_the_edge_order():
    df = pl.DataFrame({"anime_id": [50, 30, 90, 70], "related_id": [30, 90, 10, 71],
                       "relation": ["sequel", "spin_off", "alternative_setting", "other"]},
                      schema={"anime_id": pl.Int32, "related_id": pl.Int32, "relation": relations.RELATION})
    ids = [90, 70, 71, 50, 30, 10]
    fr = relations.franchise_ids(ids, df, config.FRANCHISE_RELATIONS)
    assert fr.to_list() == [10, 70, 71, 10, 10, 10]
