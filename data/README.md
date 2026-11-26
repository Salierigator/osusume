# data/

Local data, written and read by the code in this repo: **[`crawler/`](../crawler/) → `raw/` → [`cleaning/`](../cleaning/) → `clean/`**.

| Path | Committed | What |
|---|---|---|
| `raw/` | no | Crawler output, the 2026 MAL snapshot: `anime_*.jsonl`, `users.csv`, `ratings/`. |
| `clean/` | no | Training input: `anime.parquet`, `anime_relations.parquet`, `users.parquet`, `ratings/`. Schemas and rules: [`cleaning/README.md`](../cleaning/README.md). |
| `eda/` | no | Per-user tables behind the survey and EDA notebooks. |
| `samples/` | yes | The schema reference. `samples/raw/` and `samples/clean/` hold the same 20 random anime and 20 random users, as rows of the `raw/` and of the `clean/` tables, with every rating of the first 5 of those users (1,209). `samples/clean/` adds, for franchises, the anime of those 20 anime's franchises and of the two franchises around one detached crossover (53 anime, with their `franchise_id`, `title` and `members`) with every relation they list (103 rows). Written by [`create_sample.py`](samples/create_sample.py). |

The tables are MAL-derived and gigabytes in size, so nothing under `raw/` or `clean/` is committed. Read schemas from `samples/` instead of opening the real files.
