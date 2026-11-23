# Crawler

Collects a fresh anime + user-rating snapshot from MyAnimeList into `data/raw/` (gitignored). The three user crawlers are rate-limited, resumable (stop with Ctrl-C, rerun to continue), and dedup-safe — they share one SQLite state file so a rerun never re-does finished work. The catalog crawler is a one-minute sweep that rewrites its file; the relations crawler appends to its file and resumes from it.

## Scripts

| Script | Source | Output | Full-crawl time |
|---|---|---|---|
| `collect_usernames.py` | scrape `users.php` (recently-online) | state DB | background, days |
| `crawl_details.py` | MAL API v2 `/anime/ranking` | `anime_mal.jsonl` | ~1 min |
| `crawl_related.py` | MAL API v2 `/anime/{id}?fields=related_anime` | `anime_relations.jsonl` | ~11 hours / 30.8k anime |
| `crawl_ratings.py` | MAL API v2 `/users/{u}/animelist` | `ratings.csv` | ~1–2 days / 300k users |
| `crawl_profiles.py` | scrape `/profile/{u}` | `users.csv` | ~4 days / 300k users |

The catalog and the ratings come from the official MAL API (500 anime or 1000 list entries per request); profiles are parsed from the profile page, which the API does not expose. `collect_usernames.py` feeds active usernames to the ratings and profile crawlers. Relations (sequel, side story, ...) need one request per anime of `anime_mal.jsonl`, and the API answers about 100 requests per two minutes: past that it holds requests for 30 to 60 seconds instead of answering 429, so `crawl_related.py` paces itself just under the limit and backs off on timeouts.

## Running a snapshot

```bash
# 1. run in the background for days (nohup/tmux) — feeds the other two:
venv/bin/python crawler/collect_usernames.py
# 2. anytime, ~1 min; then the relations of that catalog, ~11 hours (rerun to resume):
venv/bin/python crawler/crawl_details.py
venv/bin/python crawler/crawl_related.py
# 3+4. in parallel with 1, rerun until nothing is pending:
venv/bin/python crawler/crawl_ratings.py
venv/bin/python crawler/crawl_profiles.py
```

Add `--iterations 3` / `--pages 2` / `--limit 5` for a quick smoke test.

Progress and state live in `data/raw/crawl_state.sqlite`. Check it any time:

```bash
sqlite3 data/raw/crawl_state.sqlite \
  "SELECT ratings_status, profile_status, COUNT(*) FROM users GROUP BY 1,2"
```

Private (403), deleted (404), or empty accounts are marked once and never retried. Data is written before its state is marked, so a crash can leave duplicate rows — cleaning always dedups by keeping the last copy.

## Output

- `anime_mal.jsonl` — raw MAL API anime objects, one per line (`genres` holds genres, themes and demographics together); cleaning merges them into the anime table.
- `anime_relations.jsonl` — one line per anime: `id`, `status` (200, or 404 for an anime MAL has deleted), `crawled_at` and `related_anime`, the API's list of related anime with their relation type.
- `ratings.csv` — one row per user–anime: `username, anime_id, status, score, num_episodes_watched, is_rewatching, updated_at, start_date, finish_date`.
- `users.csv` — one row per user: profile fields (gender, birthday, location, joined date, favorites) plus list stats (days watched, mean score, per-status counts, …).

[`cleaning/`](../cleaning/README.md) reads these files and produces `data/clean/`.

## Tests

```bash
venv/bin/python -m pytest crawler/tests -q   # offline: saved fixtures, no network
```
