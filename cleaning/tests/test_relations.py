"""Relation loading, franchise links and franchise components, on tiny hand-made relation graphs."""
import json

import polars as pl
import pytest

from cleaning import relations


def write(tmp_path, rows):
    path = tmp_path / "anime_relations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def rel(aid, kind):
    return {"node": {"id": aid, "title": f"anime {aid}"}, "relation_type": kind}


def catalog(ids, short=(), half=(), movies=(), members=None):
    """Full-length TV series by default; `short` are commercials, `half` TV series of 12-minute episodes."""
    return pl.DataFrame({"anime_id": ids,
                         "media_type": ["cm" if a in short else "movie" if a in movies else "tv" for a in ids],
                         "episode_minutes": [0.5 if a in short else 12.0 if a in half else 24.0 for a in ids],
                         "members": [(members or {}).get(a, 100) for a in ids]},
                        schema={"anime_id": pl.Int32, "media_type": pl.String, "episode_minutes": pl.Float32,
                                "members": pl.Int32})


def franchises(rows, anime):
    """Franchises with more than one anime, as sorted lists, and what the rules found."""
    df = pl.DataFrame(rows, schema={"anime_id": pl.Int32, "related_id": pl.Int32, "relation": relations.RELATION},
                      orient="row")
    links, found = relations.franchise_links(df, anime)
    ids = anime["anime_id"].to_list()
    groups = {}
    for a, f in zip(ids, relations.franchise_ids(ids, df.with_columns(links))):
        groups.setdefault(f, []).append(a)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1), found


def both(a, b):
    return [(a, b, "side_story"), (b, a, "parent_story")]


def chain(*ids):
    return [r for a, b in zip(ids, ids[1:]) for r in ((a, b, "sequel"), (b, a, "prequel"))]


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
    links, found = relations.franchise_links(df, catalog(CATALOG))
    assert links.to_list() == [True, False, True, True, False]
    assert found == {"crossovers_detached": 0, "short_bridges_detached": 0, "links_cut": 0}
    fr = relations.franchise_ids(CATALOG, df.with_columns(links))
    # 1-2 sequel, 3 -> 1 side story listed on one side only; 1-4 character and 5-6 other do not count
    assert dict(zip(CATALOG, fr.to_list())) == {1: 1, 2: 1, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    assert fr.dtype == pl.Int32


def test_franchise_id_is_the_smallest_id_whatever_the_edge_order():
    df = pl.DataFrame({"anime_id": [50, 30, 90, 70], "related_id": [30, 90, 10, 71],
                       "franchise_link": [True, True, True, False]},
                      schema={"anime_id": pl.Int32, "related_id": pl.Int32, "franchise_link": pl.Boolean})
    ids = [90, 70, 71, 50, 30, 10]
    assert relations.franchise_ids(ids, df).to_list() == [10, 70, 71, 10, 10, 10]


def test_parents_need_a_direction():
    rows = [*both(1, 2), (3, 4, "spin_off"), (4, 3, "spin_off"), (5, 6, "spin_off"), (7, 8, "parent_story"),
            (9, 10, "sequel")]
    assert relations.parents(rows) == {2: {1: True}, 6: {5: False}, 7: {8: False}}


def test_crossover_keeps_the_parent_that_lists_it_back():
    # 1-2 and 3-4 are two franchises; 5 is a side story of 1 (both pages say so) and a spin-off of 3 (only 3 says
    # so), and 3 has more members; 6 follows 5. 7 hangs below 1 and 2, and 8 below 1 and 7: fine within a franchise
    rows = (chain(1, 2) + chain(3, 4) + both(1, 5) + [(3, 5, "spin_off")] + chain(5, 6) + both(1, 7) + both(2, 7)
            + both(1, 8) + both(7, 8))
    found = franchises(rows, catalog(range(1, 9), members={3: 1000}))
    assert found == ([[1, 2, 5, 6, 7, 8], [3, 4]], {"crossovers_detached": 1, "short_bridges_detached": 0,
                                                    "links_cut": 1})


def test_crossover_without_a_mutual_parent_goes_to_the_most_members():
    rows = chain(1, 2) + chain(3, 4) + [(1, 5, "spin_off"), (3, 5, "spin_off")]
    assert franchises(rows, catalog(range(1, 6), members={3: 1000}))[0] == [[1, 2], [3, 4, 5]]


def test_crossover_loses_its_prequel():
    # 5 crosses 1-2 and 3-4, and lists 6 (a third franchise, 6-7) as its prequel
    rows = chain(1, 2) + chain(3, 4) + chain(6, 7) + both(1, 5) + both(3, 5) + chain(6, 5)
    assert franchises(rows, catalog(range(1, 8), members={1: 1000}))[0] == [[1, 2, 5], [3, 4], [6, 7]]


def test_crossovers_cannot_vouch_for_each_other():
    # 5 and 6 both cross 1-2 and 3-4: leaving out one at a time, the other still joins them
    rows = chain(1, 2) + chain(3, 4) + both(1, 5) + both(3, 5) + both(1, 6) + both(3, 6)
    groups, found = franchises(rows, catalog(range(1, 7), members={1: 1000}))
    assert groups == [[1, 2, 5, 6], [3, 4]] and found["crossovers_detached"] == 2


def test_spin_off_filed_both_ways_still_joins():
    rows = chain(1, 2) + chain(3, 4) + [(1, 5, "spin_off"), (5, 1, "spin_off"), (3, 5, "spin_off"),
                                        (5, 3, "spin_off")]
    assert franchises(rows, catalog(range(1, 6)))[0] == [[1, 2, 3, 4, 5]]


def test_short_entries_between_two_full_length_series_stand_apart():
    # commercials 5-6 (a sequel pair) link the series 1-2 and 3-4
    rows = chain(1, 2) + chain(3, 4) + chain(5, 6) + [(1, 5, "alternative_version"), (4, 6, "side_story")]
    groups, found = franchises(rows, catalog(range(1, 7), short={5, 6}))
    assert groups == [[1, 2], [3, 4], [5, 6]]
    assert found == {"crossovers_detached": 0, "short_bridges_detached": 1, "links_cut": 2}


@pytest.mark.parametrize("kind", ["movies", "half"])
def test_short_entries_join_when_one_side_has_no_full_length_series(kind):
    # 3-4 are movies, or TV series of 12-minute episodes
    rows = chain(1, 2) + chain(3, 4) + [(1, 5, "alternative_version"), (4, 5, "side_story")]
    anime = catalog(range(1, 6), short={5}, **{kind: {3, 4}})
    assert franchises(rows, anime)[0] == [[1, 2, 3, 4, 5]]
