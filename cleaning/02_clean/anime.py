"""Write the clean anime table.

Reads data/raw/anime_mal.jsonl and data/raw/anime_jikan.jsonl (merge rules: cleaning/catalog.py).
Writes data/clean/anime.parquet, one row per anime_id.

Usage: python -m cleaning.02_clean.anime
"""
from __future__ import annotations

from cleaning import catalog, config


def main() -> None:
    anime = catalog.build_anime()
    config.CLEAN.mkdir(parents=True, exist_ok=True)
    anime.write_parquet(config.CLEAN_ANIME, compression="zstd")
    print(f"[done] {anime.height:,} anime ({(~anime['on_mal']).sum()} no longer on MAL) "
          f"-> {config.CLEAN_ANIME.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
