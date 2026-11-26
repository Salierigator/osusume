"""Write the clean anime table and the relations between anime.

Reads data/raw/anime_mal.jsonl and data/raw/anime_jikan.jsonl (merge rules: cleaning/catalog.py) and
data/raw/anime_relations.jsonl (cleaning/relations.py). Writes data/clean/anime.parquet, one row per
anime_id with its franchise_id, data/clean/anime_relations.parquet, each relation marked as a franchise link or
not, and the "relations" section of data/clean/cleaning_report.json.

Usage: python -m cleaning.02_clean.anime
"""
from __future__ import annotations

import json

from cleaning import catalog, config, relations


def main() -> None:
    anime = catalog.build_anime()
    ids = anime["anime_id"].to_list()
    edges, counts = relations.load_relations(config.RAW_ANIME_RELATIONS, ids)
    links, found = relations.franchise_links(edges, anime)
    counts.update(found)
    edges = edges.with_columns(links)
    anime = anime.with_columns(relations.franchise_ids(ids, edges))
    config.CLEAN.mkdir(parents=True, exist_ok=True)
    anime.write_parquet(config.CLEAN_ANIME, compression="zstd")
    edges.write_parquet(config.CLEAN_ANIME_RELATIONS, compression="zstd")

    report = json.loads(config.CLEAN_REPORT.read_text()) if config.CLEAN_REPORT.exists() else {}
    report["relations"] = dict(counts)
    config.CLEAN_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[done] {anime.height:,} anime ({(~anime['on_mal']).sum()} no longer on MAL) in "
          f"{anime['franchise_id'].n_unique():,} franchises -> {config.CLEAN_ANIME.relative_to(config.ROOT)}")
    print(f"[done] {edges.height:,} relations -> {config.CLEAN_ANIME_RELATIONS.relative_to(config.ROOT)}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
