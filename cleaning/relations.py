"""Relations between anime (sequel, side story, ...) and the franchises they imply.

Relations stay as MAL lists them: one directed row per (anime, related anime, type). A franchise is a connected
component over the franchise links: relations of config.FRANCHISE_RELATIONS, each edge taken both ways because MAL
sometimes lists a relation on one side only, minus what two rules cut so that one entry cannot merge two franchises:

- Detach crossovers. P is a parent of X when X lists P as parent_story, or P lists X as side_story / spin_off and X
  does not list P the same way (MAL often files an entry and its spin-off as spin-offs of each other, which says
  nothing about direction). Entries with two or more parents are left out together, then put back while their
  parents still meet without them; the rest are crossovers. A crossover keeps one parent, one whose page lists it
  back and then the one with the most members, and loses its links to the other parents and to its prequels.
- Detach short bridges. Short entries (config.SHORT_MEDIA, or episodes under config.SHORT_MINUTES) linked together
  lose their links to longer entries when those reach two or more franchises with a full-length TV series
  (episodes of config.FULL_LENGTH_MINUTES or more).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

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


def components(nodes, pairs) -> dict:
    """node -> the smallest node of its connected component over the undirected `pairs`."""
    root = {n: n for n in nodes}

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)    # the root stays the smallest id
    return {n: find(n) for n in nodes}


def parents(rows) -> dict:
    """{X: {P: mutual}} for the parents P of each X among directed (anime_id, related_id, relation) rows; mutual
    when X lists P as parent_story and P lists X as side_story / spin_off."""
    down = {(a, b) for a, b, t in rows if t in ("side_story", "spin_off")}
    up = {(a, b) for a, b, t in rows if t == "parent_story"}
    out = defaultdict(dict)
    for x, p in up | {(b, a) for a, b in down if (b, a) not in down}:
        out[x][p] = (x, p) in up and (p, x) in down
    return out


def detach_crossovers(nodes, rows, members) -> tuple[set, int]:
    """The links the crossovers among `rows` lose (pairs of ids), and how many crossovers there are."""
    par = parents(rows)
    links = {x: {frozenset((x, p)) for p in ps} for x, ps in par.items() if len(ps) >= 2}
    base = {frozenset(r[:2]) for r in rows}.difference(*links.values())
    found = set(links)
    while True:
        root = components(nodes, base.union(*(links[x] for x in set(links) - found)))
        still = {x for x in found if len({root[p] for p in par[x]}) > 1}
        if still == found:
            break
        found = still
    prequels = defaultdict(set)
    for a, b, t in rows:
        if t == "prequel":
            prequels[a].add(b)
        elif t == "sequel":
            prequels[b].add(a)
    cut = set()
    for x in found:
        mutual = [p for p, m in par[x].items() if m]
        home = max(mutual or par[x], key=lambda p: (members[p], -p))
        cut |= {frozenset((x, p)) for p in set(par[x]) | prequels[x] if p != home}
    return cut, len(found)


def detach_short_bridges(nodes, pairs, short, full_length) -> tuple[set, int]:
    """The links between short and longer entries to cut (pairs of ids), and how many short clusters lose them."""
    long_root = components(nodes, [e for e in pairs if not e & short])
    short_root = components(nodes, [e for e in pairs if e <= short])
    anchored = {long_root[n] for n in full_length}
    reach = defaultdict(set)
    for e in pairs:
        if len(e & short) == 1:
            reach[short_root[next(iter(e & short))]].add(e)
    bridges = [links for links in reach.values()
               if len({long_root[n] for e in links for n in e - short} & anchored) >= 2]
    return set().union(*bridges), len(bridges)


def franchise_links(relations: pl.DataFrame, anime: pl.DataFrame) -> tuple[pl.Series, Counter]:
    """For each row of `relations`, whether it is a franchise link (module docstring). `anime` gives anime_id,
    media_type, episode_minutes and members. Also returns what the two rules found."""
    rows = relations.select("anime_id", "related_id", pl.col("relation").cast(pl.String)).rows()
    strong = [r for r in rows if r[2] in config.FRANCHISE_RELATIONS]
    nodes = anime["anime_id"].to_list()
    cut, crossovers = detach_crossovers(nodes, strong, dict(zip(nodes, anime["members"].fill_null(0))))
    short = set(anime.filter(pl.col("media_type").is_in(config.SHORT_MEDIA)
                             | (pl.col("episode_minutes") < config.SHORT_MINUTES))["anime_id"])
    full_length = set(anime.filter((pl.col("media_type") == "tv")
                                   & (pl.col("episode_minutes") >= config.FULL_LENGTH_MINUTES))["anime_id"])
    bridged, bridges = detach_short_bridges(nodes, {frozenset(r[:2]) for r in strong} - cut, short, full_length)
    cut |= bridged
    links = pl.Series("franchise_link", [t in config.FRANCHISE_RELATIONS and frozenset((a, b)) not in cut
                                         for a, b, t in rows])
    return links, Counter(crossovers_detached=crossovers, short_bridges_detached=bridges,
                          links_cut=len(strong) - links.sum())


def franchise_ids(anime_ids, relations: pl.DataFrame) -> pl.Series:
    """For each id of `anime_ids`, in order: the smallest anime_id of its connected component over the rows of
    `relations` whose franchise_link is true."""
    root = components(anime_ids, relations.filter("franchise_link").select("anime_id", "related_id").rows())
    return pl.Series("franchise_id", [root[a] for a in anime_ids], dtype=pl.Int32)
