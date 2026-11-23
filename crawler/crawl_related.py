"""Crawl the relations of every catalog anime (sequel, prequel, side story, ...) from the MAL API v2.

Only /anime/{id}?fields=related_anime returns them; the list endpoints ignore that field, so this
makes one request per anime. Ids come from data/raw/anime_mal.jsonl (crawl_details.py).

Output: data/raw/anime_relations.jsonl, one line per anime: {"id", "status" (200, or 404 for an id
MAL has deleted), "crawled_at", "related_anime"}, where related_anime is the API's list as returned.
Lines are appended as answers arrive, so a rerun skips the ids already written and retries the rest.

MAL answers about 100 requests per two minutes, whatever the pace; past that it holds every open
request for 30 to 60 seconds instead of answering 429. So the workers share one rate limit just under
that, and a timeout, network error, 429 or 5xx counts as throttling: the id goes to the back of the
queue, every worker pauses, and the rate drops by a fifth, then creeps back up to --rate. Other
errors are retried without a pause, and the run stops early when many requests fail in a row (a
refused API key, a blocked IP).

Exit code: 0 when every id is answered, 2 when some are still pending (rerun), 1 when it stopped
on failures.

Usage: python crawler/crawl_related.py [--out PATH] [--limit N] [--workers N] [--rate R]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

import common

IDS_FROM = common.RAW / "anime_mal.jsonl"
OUT = common.RAW / "anime_relations.jsonl"
DONE = (200, 404)


def throttled(status) -> bool:
    """A failure that means 'slow down': network error or timeout (an exception name), 429, 5xx."""
    return isinstance(status, str) or status == 429 or status >= 500


class Limiter:
    """Spaces requests from all threads `interval` seconds apart. slow_down() pauses everyone and
    cuts the rate by a fifth (once per pause, however many requests failed together); each
    success then shrinks the interval by 0.1% back towards the starting value."""

    def __init__(self, rate: float):
        self.base = self.interval = 1.0 / rate
        self.next = self.paused_until = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)

    def slow_down(self, pause: float) -> None:
        with self.lock:
            now = time.monotonic()
            if now >= self.paused_until:             # first failure of this stall
                self.interval = min(self.interval * 1.25, 10.0)
                self.paused_until = now + pause
            self.next = max(self.next, self.paused_until)

    def recover(self) -> None:
        with self.lock:
            self.interval = max(self.base, self.interval * 0.999)


def make_fetch(client_id: str, timeout: float):
    """fetch(anime_id) -> (status, related_anime). status is the HTTP code, or the exception name
    for a network error; related_anime is None unless status is 200 or 404."""
    local = threading.local()

    def fetch(aid):
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.headers["X-MAL-CLIENT-ID"] = client_id
        try:
            r = local.session.get(f"{common.MAL_BASE}/anime/{aid}",
                                  params={"fields": "related_anime"}, timeout=(5, timeout))
            if r.status_code == 200:
                return 200, r.json().get("related_anime", [])
        except (requests.RequestException, ValueError) as e:   # ValueError: truncated JSON body
            return type(e).__name__, None
        return r.status_code, ([] if r.status_code == 404 else None)

    return fetch


def load_done(out: Path) -> set:
    """Ids already answered in `out`. A partly written last line (killed mid-write, or a copy
    taken while the file grew) is cut off so that new lines start clean."""
    if not out.exists():
        return set()
    data = out.read_bytes()
    end = data.rfind(b"\n") + 1
    if end < len(data):
        with open(out, "r+b") as f:
            f.truncate(end)
    return {row["id"] for row in map(json.loads, data[:end].splitlines()) if row["status"] in DONE}


def crawl(ids, out: Path, fetch, *, workers: int = 2, limiter: Limiter | None = None,
          limit: int | None = None, max_attempts: int = 8, max_fail_streak: int = 50,
          pause: float = 30.0, log_every: float = 60.0) -> dict:
    """Fetch every id not yet answered in `out` (the first `limit` of them if given), appending one
    line per answer. Returns written, pending (ids still unanswered), statuses (every attempt),
    aborted and seconds."""
    limiter = limiter or Limiter(0.8)
    done = load_done(out)
    queue = deque((aid, 0) for aid in ids if aid not in done)
    if limit is not None:
        queue = deque(list(queue)[:limit])
    total = len(queue)
    print(f"[start] {len(done):,} ids already in {out.name}, {total:,} to fetch; "
          f"{workers} workers, {1 / limiter.interval:.1f} req/s", flush=True)

    statuses, pending = Counter(), []
    written = fail_streak = 0
    aborted = False
    start = last_log = time.monotonic()
    marks = deque([(start, 0)])

    def task(aid):
        limiter.wait()
        return fetch(aid)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        running = {}
        while queue or running:
            while queue and len(running) < 2 * workers and not (common.STOP or aborted):
                aid, tries = queue.popleft()
                running[pool.submit(task, aid)] = (aid, tries)
            if not running:
                break
            finished, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in finished:
                aid, tries = running.pop(fut)
                status, related = fut.result()
                statuses[status] += 1
                if status in DONE:
                    f.write(json.dumps({"id": aid, "status": status, "crawled_at": common.now_iso(),
                                        "related_anime": related}, ensure_ascii=False) + "\n")
                    written += 1
                    fail_streak = 0
                    limiter.recover()
                    continue
                fail_streak += 1
                if throttled(status):
                    limiter.slow_down(pause)
                if tries + 1 < max_attempts:
                    queue.append((aid, tries + 1))
                else:
                    pending.append(aid)
                if fail_streak >= max_fail_streak and not aborted:
                    aborted = True
                    print(f"[abort] {fail_streak} failed requests in a row, last: id {aid} -> {status}",
                          flush=True)
            f.flush()
            now = time.monotonic()
            if now - last_log >= log_every:
                marks.append((now, written))
                while now - marks[0][0] > 300 and len(marks) > 2:
                    marks.popleft()
                rate = (written - marks[0][1]) / (now - marks[0][0])
                eta = (total - written) / rate / 3600 if rate else float("inf")
                print(f"[{time.strftime('%H:%M:%S')}] {written:,}/{total:,} written "
                      f"({written / max(total, 1):.1%}), {rate:.2f}/s over the last "
                      f"{now - marks[0][0]:.0f} s, ETA {eta:.2f} h; retry queue "
                      f"{sum(t > 0 for _, t in queue)}, interval {limiter.interval:.3f} s; "
                      f"attempts {dict(statuses)}", flush=True)
                last_log = now
    pending.extend(aid for aid, _ in queue)
    return {"written": written, "pending": pending, "statuses": dict(statuses), "aborted": aborted,
            "seconds": time.monotonic() - start}


def main():
    ap = argparse.ArgumentParser(description="Crawl anime relations from the MAL API v2.")
    ap.add_argument("--out", type=Path, default=OUT, help="output JSONL (default data/raw/anime_relations.jsonl)")
    ap.add_argument("--limit", type=int, help="fetch only the first N ids not yet in --out (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rate", type=float, default=0.8, help="requests per second, all workers together")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for an answer")
    args = ap.parse_args()

    common.install_signals()
    ids = [json.loads(line)["id"] for line in IDS_FROM.open(encoding="utf-8")]
    fetch = make_fetch(common.load_mal_client_id(), args.timeout)
    s = crawl(ids, args.out, fetch, workers=args.workers, limiter=Limiter(args.rate), limit=args.limit)
    print(f"[done] {s['written']:,} written in {s['seconds'] / 60:.1f} min "
          f"({s['written'] / max(s['seconds'], 1e-9):.2f}/s); attempts {s['statuses']}; "
          f"{len(s['pending'])} pending {s['pending'][:10]}", flush=True)
    sys.exit(1 if s["aborted"] else 2 if s["pending"] else 0)


if __name__ == "__main__":
    main()
