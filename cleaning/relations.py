"""Relations between anime (sequel, side story, ...) and the franchises they imply.

Relations stay as MAL lists them: one directed row per (anime, related anime, type). A franchise is
a connected component of the relation graph over config.FRANCHISE_RELATIONS, each edge taken both
ways, because MAL sometimes lists a relation on one side only.
"""
from __future__ import annotations

import json
from collections import Counter

import polars as pl

from cleaning import config

RELATION = pl.Enum(config.RELATION_TYPES)


def load_relations(path, anime_ids) -> tuple[pl.DataFrame, Counter]:
    """The raw relation file -> one row per (anime_id, related_id, relation) with both ids in
    `anime_ids`, sorted. A repeated id keeps its last line. Also returns what each rule removed."""
    lines = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        lines[r["id"]] = r
    counts = Counter(lines=len(lines), **{f"status_{s}": n for s, n in
                                          Counter(r["status"] for r in lines.values()).items()})
    edges = [(aid, rel["node"]["id"], rel["relation_type"])
             for aid, r in lines.items() for rel in r["related_anime"]]
    unknown = {t for _, _, t in edges} - set(config.RELATION_TYPES)
    if unknown:
        raise ValueError(f"relation types missing from config.RELATION_TYPES: {sorted(unknown)}")
    counts["edges_in"] = len(edges)

    known = set(anime_ids)
    df = pl.DataFrame(edges, schema={"anime_id": pl.Int32, "related_id": pl.Int32, "relation": pl.String},
                      orient="row")
    before = df.height
    df = df.filter(pl.col("anime_id").is_in(known) & pl.col("related_id").is_in(known))
    counts["dropped_unknown_anime"] = before - df.height
    before = df.height
    df = df.filter(pl.col("anime_id") != pl.col("related_id"))
    counts["dropped_self"] = before - df.height
    before = df.height
    df = df.unique()
    counts["dropped_duplicate"] = before - df.height
    counts["edges_out"] = df.height
    return df.with_columns(pl.col("relation").cast(RELATION)).sort("anime_id", "related_id", "relation"), counts


def franchise_ids(anime_ids, relations: pl.DataFrame, types) -> pl.Series:
    """For each id of `anime_ids`, in order: the smallest anime_id of its connected component in the
    graph of `relations` restricted to `types`, edges taken as undirected."""
    parent = {a: a for a in anime_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = relations.filter(pl.col("relation").cast(pl.String).is_in(list(types)))
    for a, b in pairs.select("anime_id", "related_id").iter_rows():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)    # the root stays the smallest id
    return pl.Series("franchise_id", [find(a) for a in anime_ids], dtype=pl.Int32)
