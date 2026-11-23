"""Offline tests for crawl_related, with a fake fetch that answers from a script of statuses."""
from __future__ import annotations

import json
import time

import common
import crawl_related as cr


def fake_fetch(script):
    """fetch(aid) answering from `script`: anime_id -> statuses for successive attempts (the last
    one repeats; ids not in the script answer 200)."""
    calls = []

    def fetch(aid):
        calls.append(aid)
        seq = script.get(aid, [200])
        status = seq[min(calls.count(aid), len(seq)) - 1]
        if status == 200:
            return 200, [{"node": {"id": aid + 1000, "title": "x"}, "relation_type": "sequel"}]
        return status, ([] if status == 404 else None)

    fetch.calls = calls
    return fetch


def run(tmp_path, ids, script=None, **kw):
    out = tmp_path / "relations.jsonl"
    kw.setdefault("limiter", cr.Limiter(1000.0))
    kw.setdefault("pause", 0.0)
    fetch = fake_fetch(script or {})
    summary = cr.crawl(ids, out, fetch, **kw)
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    return summary, rows, fetch.calls


def test_one_line_per_answer(tmp_path):
    summary, rows, _ = run(tmp_path, [1, 2, 3], {2: [404]})
    by_id = {r["id"]: r for r in rows}
    assert sorted(by_id) == [1, 2, 3]
    assert all(set(r) == {"id", "status", "crawled_at", "related_anime"} for r in rows)
    assert by_id[2]["status"] == 404 and by_id[2]["related_anime"] == []
    assert by_id[1]["related_anime"][0]["node"]["id"] == 1001
    assert summary["written"] == 3 and summary["pending"] == [] and not summary["aborted"]


def test_resume_skips_answered_ids_and_cuts_a_partial_line(tmp_path):
    out = tmp_path / "relations.jsonl"
    out.write_text(json.dumps({"id": 1, "status": 200, "crawled_at": "t", "related_anime": []})
                   + "\n" + '{"id": 2, "sta')
    _, rows, calls = run(tmp_path, [1, 2, 3])
    assert sorted(calls) == [2, 3]
    assert sorted(r["id"] for r in rows) == [1, 2, 3]


def test_limit_takes_the_first_unanswered_ids(tmp_path):
    _, rows, calls = run(tmp_path, [5, 6, 7, 8], limit=2)
    assert sorted(calls) == [5, 6] and sorted(r["id"] for r in rows) == [5, 6]


def test_network_errors_and_5xx_are_retried(tmp_path):
    summary, rows, calls = run(tmp_path, [1, 2], {1: ["ReadTimeout", 503, 200]})
    assert calls.count(1) == 3
    assert {r["id"]: r["status"] for r in rows} == {1: 200, 2: 200}
    assert summary["statuses"]["ReadTimeout"] == 1 and summary["statuses"][503] == 1


def test_throttling_slows_every_worker_down(tmp_path):
    for status in (429, "ReadTimeout", 503):
        limiter = cr.Limiter(1000.0)
        _, rows, calls = run(tmp_path / str(status), [1], {1: [status, 200]}, limiter=limiter)
        assert calls == [1, 1] and rows[0]["status"] == 200
        assert limiter.interval > limiter.base      # one success is not enough to recover


def test_other_errors_retry_without_slowing_down(tmp_path):
    limiter = cr.Limiter(1000.0)
    _, rows, calls = run(tmp_path, [1], {1: [403, 200]}, limiter=limiter)
    assert calls == [1, 1] and rows[0]["status"] == 200
    assert limiter.interval == limiter.base


def test_limiter_pause_and_recovery():
    lim = cr.Limiter(100.0)
    lim.slow_down(0.2)
    lim.slow_down(0.2)                              # a sibling request failing in the same stall
    assert abs(lim.interval - 0.0125) < 1e-12
    t = time.monotonic()
    lim.wait()
    assert time.monotonic() - t >= 0.19
    for _ in range(2000):
        lim.recover()
    assert lim.interval == lim.base


def test_slow_down_works_while_the_next_slot_is_booked():
    lim = cr.Limiter(1.0)
    lim.wait()                                      # books the slot one second ahead
    lim.slow_down(0.0)
    assert lim.interval == 1.25


def test_gives_up_after_max_attempts(tmp_path):
    summary, rows, calls = run(tmp_path, [1, 2, 3], {2: [503]}, max_attempts=3)
    assert calls.count(2) == 3
    assert sorted(r["id"] for r in rows) == [1, 3]
    assert summary["pending"] == [2] and not summary["aborted"]


def test_stops_when_requests_keep_failing(tmp_path):
    ids = list(range(1, 40))
    summary, rows, calls = run(tmp_path, ids, {i: [403] for i in ids}, max_fail_streak=5, workers=2)
    assert summary["aborted"] and rows == []
    assert len(calls) <= 5 + 2 * 2                  # the streak, plus requests already in flight
    assert sorted(summary["pending"]) == ids


def test_stop_signal_leaves_everything_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "STOP", True)
    summary, rows, calls = run(tmp_path, [1, 2])
    assert calls == [] and rows == [] and sorted(summary["pending"]) == [1, 2]
