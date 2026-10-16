"""Unit tests for dictdact.context: the pure terms_from_texts() clustering and the threaded
ContextReader wrapper around visible_texts().

visible_texts() touches macOS Accessibility (AppKit/ApplicationServices) and is NEVER called
for real here — every ContextReader(enabled=True) test monkeypatches
dictdact.context.visible_texts, and the enabled=False test asserts it is not even called.
"""
import time

from dictdact.context import ContextReader, terms_from_texts


# -- terms_from_texts (pure) ---------------------------------------------------------------

def test_terms_from_texts_empty_input_returns_empty_list():
    assert terms_from_texts([]) == []


def test_terms_from_texts_excludes_a_named_term():
    texts = ["Talk to Simone about Hyrox", "Hyrox again with ManyChat", "hoy no"]
    out = terms_from_texts(texts, exclude={"ManyChat"})
    assert "ManyChat" not in out
    assert set(out) == {"Hyrox", "Simone"}


def test_terms_from_texts_limit_caps_the_result_length():
    texts = ["Talk to Simone about Hyrox", "Hyrox again with ManyChat", "hoy no"]
    out = terms_from_texts(texts, exclude={"ManyChat"}, limit=1)
    assert len(out) == 1
    assert out[0] in {"Hyrox", "Simone"}


def test_terms_from_texts_orders_by_frequency_first():
    texts = ["Talk to Simone about Hyrox", "Hyrox again with ManyChat", "hoy no"]
    assert terms_from_texts(texts, exclude={"ManyChat"}) == ["Hyrox", "Simone"]
    assert terms_from_texts(texts, exclude={"ManyChat"}, limit=1) == ["Hyrox"]


# -- ContextReader --------------------------------------------------------------------------

def test_disabled_reader_never_reads_and_take_is_always_empty(monkeypatch):
    import dictdact.context as context

    def must_not_be_called(*a, **k):
        raise AssertionError("visible_texts must not be called when the reader is disabled")

    monkeypatch.setattr(context, "visible_texts", must_not_be_called)
    reader = ContextReader(enabled=False)
    reader.start()
    assert reader.take() == (None, [])


def _poll_take(reader, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    result = (None, [])
    while time.monotonic() < deadline:
        result = reader.take()
        if result != (None, []):
            return result
        time.sleep(0.01)
    return result


def test_enabled_reader_returns_the_app_name_and_is_consumed_after_take(monkeypatch):
    import dictdact.context as context

    monkeypatch.setattr(context, "visible_texts", lambda *a, **k: ("Claude", ["Simone and Hyrox here"]))
    reader = ContextReader(enabled=True)
    reader.start(exclude=set())

    name, terms = _poll_take(reader)
    assert name == "Claude"
    assert terms                              # something was extracted from the visible text
    assert set(terms) <= {"Simone", "Hyrox"}    # only real content, nothing invented

    # a take() result is consumed: the next call with nothing newly queued is the empty sentinel
    assert reader.take() == (None, [])


def test_enabled_reader_extracts_both_names_from_the_visible_text(monkeypatch):
    import dictdact.context as context

    monkeypatch.setattr(context, "visible_texts", lambda *a, **k: ("Claude", ["Simone and Hyrox here"]))
    reader = ContextReader(enabled=True)
    reader.start(exclude=set())

    result = _poll_take(reader)
    assert result == ("Claude", ["Simone", "Hyrox"])


def test_a_slow_read_never_lands_on_a_later_take(monkeypatch):
    import threading, time
    import dictdact.context as context
    gate = threading.Event()
    calls = []

    def slow_then_fast(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            gate.wait(2.0)                      # take 1's read is slow
            return ("Slack", ["Confidential Merger"])
        return ("Xcode", ["Simone here"])

    monkeypatch.setattr(context, "visible_texts", slow_then_fast)
    reader = ContextReader(enabled=True)
    reader.start(exclude=set())                 # take 1
    reader.start(exclude=set())                 # take 2, right after
    assert _poll_take(reader) == ("Xcode", ["Simone"])
    gate.set()                                  # take 1's read finishes now…
    time.sleep(0.2)
    assert reader.take() == (None, [])          # …and is dropped, not handed to a later take


def test_disabling_the_reader_clears_a_pending_result(monkeypatch):
    import time
    import dictdact.context as context
    monkeypatch.setattr(context, "visible_texts", lambda *a, **k: ("1Password", ["Vault Master"]))
    reader = ContextReader(enabled=True)
    reader.start(exclude=set())
    time.sleep(0.1)                             # a result is pending and was never taken
    reader.enabled = False
    reader.start(exclude=set())                 # a new take with the feature off
    assert reader.take() == (None, [])
