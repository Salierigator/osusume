"""Account rules on a hand-made statistics table."""
import polars as pl

from cleaning import accounts, config


def test_flag_accounts():
    max_rows, max_minutes = config.MAX_LIST_SIZE, config.MAX_WATCH_DAYS * 1440.0
    # (n_rows, watch_minutes) -> (no_valid_rows, list_too_large, watch_time_impossible)
    cases = [
        ((50, 900.0), (False, False, False)),
        ((None, None), (True, False, False)),    # null when no row passed the row rules
        ((0, 0.0), (True, False, False)),
        ((max_rows, 900.0), (False, False, False)),
        ((max_rows + 1, 900.0), (False, True, False)),
        ((50, max_minutes), (False, False, False)),
        ((50, max_minutes + 1), (False, False, True)),
    ]
    stats = pl.DataFrame([row for row, _ in cases], schema={"n_rows": pl.UInt32, "watch_minutes": pl.Float64},
                         orient="row")
    out = accounts.flag_accounts(stats)

    flags = out.select("rule_no_valid_rows", "rule_list_too_large", "rule_watch_time_impossible")
    assert flags.rows() == [expected for _, expected in cases]
    assert out["dropped"].to_list() == [any(expected) for _, expected in cases]
