# Cleaning

Turns the crawler's snapshot in `data/raw/` into three typed tables in `data/clean/`, and documents the data with a survey and an EDA. Tables are written under `data/` (local, not committed); scripts, notebooks with their figures and the small aggregate tables live here.

```
cleaning/
├── config.py        # paths, row rules, account thresholds: every number a rule depends on
├── rules.py         # row fixes for ratings + the streaming reader (never splits a user)
├── catalog.py       # merges the MAL and Jikan catalogs into one anime table
├── relations.py     # relations between anime and the franchises they form
├── accounts.py      # which accounts are dropped whole
├── plotting.py      # matplotlib style shared by the notebooks
├── 01_survey/       # measure before deciding
│   ├── account_stats.py     # raw ratings -> one row of statistics per account
│   └── thresholds.ipynb     # distributions behind the account rules
├── 02_clean/        # data/raw -> data/clean
│   ├── anime.py  users.py  ratings.py
│   └── verify.py            # invariants of data/clean
├── 03_eda/          # look at the clean data
│   ├── aggregates.py        # clean tables -> small aggregate tables
│   ├── tables/              # those tables (committed)
│   └── anime.ipynb  users.ipynb  ratings.ipynb  time.ipynb  correlations.ipynb
└── tests/
```

Scripts do the heavy work and stream the ratings a few row groups at a time (3 to 6 GB of memory and 4 to 7 minutes per pass over 554M rows on a laptop). Notebooks only read the small tables those scripts wrote, so they run in seconds.

## Running

From the repo root, in this order:

```bash
python -m cleaning.01_survey.account_stats   # data/eda/survey/account_stats.parquet
python -m cleaning.02_clean.anime            # data/clean/anime.parquet, anime_relations.parquet
python -m cleaning.02_clean.users            # data/clean/users.parquet
python -m cleaning.02_clean.ratings          # data/clean/ratings/part-NN.parquet
python -m cleaning.02_clean.verify           # must end with "all checks passed"
python -m cleaning.03_eda.aggregates         # cleaning/03_eda/tables/ + data/eda/
jupyter nbconvert --to notebook --execute --inplace cleaning/01_survey/*.ipynb cleaning/03_eda/*.ipynb
python -m pytest cleaning/tests -q
```

`account_stats`, `ratings` and `aggregates` take `--max-groups N` for a quick run on the first N row groups.

## Output

| Table | Rows | Key | Content |
|---|---|---|---|
| `anime.parquet` | 30,801 | `anime_id` | titles, format, source, age rating, episodes, dates, genres / themes / demographics, studios, producers, synopsis, MAL statistics |
| `users.parquet` | 1,907,258 | `user_id` | username, crawl time, profile (gender, birth year, location, join date, favorites), list counts per status |
| `ratings/` | 548,928,922 | `user_id`, `anime_id` | status, score (0 = not scored), episodes watched, rewatching flag, `updated_at`, start and finish dates |

`user_id` is the user's position in crawl order. Ratings are sorted by `(user_id, updated_at)`, a user never spans two files, and row-group statistics on `user_id` make reading a range of users cheap.

Usernames were collected in two ways: by polling MAL's recently-online page during the crawl, and from two older username lists, from 2022 and 2025. The users are therefore a mix of accounts active during the crawl and accounts that stopped long before it: 40% have no list edit on or after 2025-07-01.

MAL statistics in the anime table (`score`, `members`, `favorites`, `rank`) were crawled after every rating, so a model evaluated on a time split must not use them as features of the test period.

## Rules

**Rows** (`rules.py`), out of 553,776,843 raw rating rows:

| Rule | Rows |
|---|---|
| dropped: no status | 226,323 |
| dropped: anime not in the catalog | 400 |
| dropped: repeated (user, anime), latest edit kept | 13 |
| `updated_at` outside 2004 .. end of crawl → null | 3,999 |
| `start_date` / `finish_date` partial or out of range → null | 1,421,969 / 1,288,832 |
| `episodes_watched` above the longest anime → null | 7,604 |

**Accounts** (`accounts.py`, thresholds in `config.py`, evidence in `01_survey/thresholds.ipynb`): an account is dropped whole when its list cannot be one person's viewing history.

| Rule | Accounts |
|---|---|
| more than 10,000 list entries (the catalog has 30.8k anime) | 208 |
| more than 1,000 days of total watch time | 229 |
| no valid row left | 7 |
| any of the above | 369, holding 4,621,185 rows (0.83%) |

**Users**: the profile file is append-only, so the last row of a username wins. Birth years outside 1940–2016 become null. Favorites that name an anime outside the catalog are removed, and the favorites of 61 users whose profile re-crawl failed are blanked.

**Anime**: one row per id from the MAL API file; the Jikan file supplies producers, the genre / theme / demographic split and 25 anime that were later deleted from MAL but still appear in lists (`on_mal = false`). Partial start dates are padded to the first day and marked in `start_precision`.

**Relations and franchises** (`relations.py`, relation types in `config.py`): `anime_relations.parquet` keeps every relation MAL lists (sequel, prequel, side story, summary, alternative version, spin-off, ...) as one directed row, minus relations to anime outside the catalog, relations of an anime to itself and repeats. `franchise_id` in the anime table is the smallest `anime_id` of a connected component over every relation type except `character` and `other`, each edge taken both ways since MAL sometimes lists a relation on one side only. `character` and `other` are left out because they also join unrelated works: crossovers such as Isekai Quartet, collaboration ads, music videos shared by several shows. The rest keeps a franchise together, e.g. Attack on Titan (20 entries: seasons, OVAs, recap movies, the Junior High spin-off) or Fate (38). Known limits: a crossover special filed as a side story still joins two franchises (Detective Conan and Lupin III), and adaptations MAL links only through `other` stay apart (the later Fate/Grand Order series).

## What is not filtered here

Small lists and rarely watched anime stay in `data/clean`. How many interactions a user or an anime needs is a modelling choice that differs between training, evaluation and serving, so each model applies its own minimum when it builds a split. For reference, 9.1% of users have fewer than 10 watched anime and 14.1% fewer than 20.

Bulk-imported lists, users who never score and users who give one score to everything are also kept: their lists still say what they watched.

## Notebooks

| Notebook | Question |
|---|---|
| `01_survey/thresholds.ipynb` | Which accounts cannot be real, and where to cut |
| `03_eda/anime.ipynb` | What the catalog contains and how activity spreads over it |
| `03_eda/users.ipynb` | Who the users are, profile coverage, list sizes, scoring habits, taste by gender |
| `03_eda/ratings.ipynb` | Status and score distributions, progress on dropped anime, optional fields |
| `03_eda/time.ipynb` | What `updated_at` supports: date splits, recency, watch order |
| `03_eda/correlations.ipynb` | What moves together across users, anime, genres and pairs of anime |
