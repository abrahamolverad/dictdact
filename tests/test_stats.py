from datetime import date

from dictdact.history import Block
from dictdact.stats import Stats, build


def blk(ts: str, text: str, seconds: float = 1.0) -> Block:
    return Block(id=ts, ts=ts, text=text, raw=text, duration_s=seconds)


def test_words_and_seconds_land_on_their_own_day():
    snap = build([blk("2026-09-01T10:00:00", "one two three", 4.0),
                  blk("2026-09-01T18:00:00", "four five", 6.0),
                  blk("2026-09-03T09:00:00", "six", 2.0)], today="2026-09-03")
    assert [(d.date, d.words, d.takes, d.seconds) for d in snap.days] == [
        ("2026-09-01", 5, 2, 10.0), ("2026-09-02", 0, 0, 0.0), ("2026-09-03", 1, 1, 2.0)]
    assert snap.words == 6 and snap.takes == 3


def test_a_silent_day_is_a_zero_bar_not_a_missing_one():
    snap = build([blk("2026-09-01T10:00:00", "a"), blk("2026-09-05T10:00:00", "b")], today="2026-09-05")
    assert len(snap.days) == 5                      # the chart must show the gap, not close it
    assert [d.words for d in snap.days] == [1, 0, 0, 0, 1]
    assert snap.days_used == 2 and snap.days_since_first == 5


def test_today_extends_the_span_even_with_nothing_said_today():
    snap = build([blk("2026-09-01T10:00:00", "a b c")], today="2026-09-04")
    assert snap.days[-1].date == "2026-09-04" and snap.days[-1].words == 0
    assert snap.today.words == 0 and snap.today.date == "2026-09-04"
    assert snap.words_per_day == 3 / 4              # a quiet day drags the average down, honestly


def test_words_per_day_and_best_day():
    snap = build([blk("2026-09-01T10:00:00", "a b"), blk("2026-09-02T10:00:00", "c d e f")], today="2026-09-02")
    assert snap.words_per_day == 3.0
    assert snap.best.date == "2026-09-02" and snap.best.words == 4


def test_empty_history_has_no_days_and_no_division_by_zero():
    snap = build([], today="2026-09-02")
    assert snap.days == [] and snap.words_per_day == 0.0 and snap.first_day is None
    assert snap.today.date == "2026-09-02" and snap.today.words == 0
    assert snap.as_dict()["totals"]["best"] is None


def test_a_line_without_a_real_timestamp_is_skipped_not_crashed():
    snap = build([blk("", "ghost"), blk("not-a-date", "ghost"), blk("2026-09-02T10:00:00", "real one")],
                 today="2026-09-02")
    assert snap.words == 2 and len(snap.days) == 1


def test_a_future_timestamp_still_gets_a_bar():
    snap = build([blk("2026-09-01T10:00:00", "a"), blk("2026-09-04T10:00:00", "b")], today="2026-09-02")
    assert snap.days[-1].date == "2026-09-04"       # clock skew must not produce a negative span
    assert snap.days_since_first == 4


class FileHistory:
    """The real History shape the cache keys on: a path plus a read."""

    def __init__(self, path):
        self.path = path
        self.reads = 0

    def all(self):
        self.reads += 1
        out = []
        for line in self.path.read_text().splitlines():
            ts, text = line.split("|", 1)
            out.append(blk(ts, text))
        return out


def test_the_file_is_read_once_until_it_changes(tmp_path):
    f = tmp_path / "history.jsonl"
    f.write_text("2026-09-01T10:00:00|hello there\n")
    h = FileHistory(f)
    s = Stats(h)
    for _ in range(5):
        assert s.load("2026-09-01")[1].words == 2
    assert h.reads == 1                              # /api/state polls once a second

    f.write_text("2026-09-01T10:00:00|hello there\n2026-09-01T11:00:00|and one more\n")
    assert s.load("2026-09-01")[1].words == 5
    assert h.reads == 2


def test_the_cache_rolls_over_at_midnight(tmp_path):
    f = tmp_path / "history.jsonl"
    f.write_text("2026-09-01T23:59:00|late night words\n")
    s = Stats(FileHistory(f))
    assert s.load("2026-09-01")[1].today.words == 3
    assert s.load("2026-09-02")[1].today.words == 0   # same file, new day
    assert s.load("2026-09-02")[1].days[-1].date == "2026-09-02"


def test_a_history_without_a_path_is_never_cached(tmp_path):
    class Fake:
        def __init__(self): self.blocks = []
        def all(self): return list(self.blocks)

    fake = Fake()
    s = Stats(fake)
    assert s.load("2026-09-01")[1].words == 0
    fake.blocks.append(blk("2026-09-01T10:00:00", "now there is one"))
    assert s.load("2026-09-01")[1].words == 4        # tests' fakes must stay live


def test_returned_blocks_cannot_corrupt_the_cache(tmp_path):
    f = tmp_path / "history.jsonl"
    f.write_text("2026-09-01T10:00:00|one two\n")
    s = Stats(FileHistory(f))
    blocks, _ = s.load("2026-09-01")
    blocks.clear()
    assert len(s.load("2026-09-01")[0]) == 1


def test_snapshot_serialises_for_the_chart():
    d = build([blk("2026-09-01T10:00:00", "a b", 90.0)], today="2026-09-01").as_dict()
    assert d["days"] == [{"date": "2026-09-01", "takes": 1, "words": 2, "seconds": 90.0}]
    assert d["totals"]["words"] == 2 and d["totals"]["days_used"] == 1
    assert d["today"]["words"] == 2


def test_a_real_looking_day_counts_the_pasted_text_not_the_raw():
    b = Block(id="x", ts="2026-09-01T10:00:00", text="two words", raw="two words spoken badly here")
    assert build([b], today="2026-09-01").words == 2
