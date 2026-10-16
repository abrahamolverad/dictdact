"""The keep-warm machinery: scheduling logic, touch hooks, config knobs."""
import queue
import threading
import time
from pathlib import Path

from dictdact.config import Config
from dictdact.format import MLXFormatter, NoFormatter, OllamaFormatter, RulesFormatter
from dictdact.pipeline import Pipeline
from dictdact.warmth import HOTKEY_PREWARM_IDLE_S, warm_due


def test_warm_due_only_after_the_interval():
    assert warm_due(last_use_ts=0.0, now=45.0, interval_s=45.0)
    assert not warm_due(last_use_ts=10.0, now=45.0, interval_s=45.0)
    assert warm_due(last_use_ts=10.0, now=100.0, interval_s=45.0)


def test_warm_due_disabled():
    assert not warm_due(0.0, 1e9, 0)
    assert not warm_due(0.0, 1e9, -5)
    assert not warm_due(0.0, 1e9, None)


def test_hotkey_prewarm_idle_is_shorter_than_default_heartbeat():
    assert 0 < HOTKEY_PREWARM_IDLE_S < Config().keep_warm_s


def test_config_roundtrip_keeps_the_warm_knobs(tmp_path: Path):
    cfg = Config()
    cfg.keep_warm_s = 0
    cfg.wire_memory = False
    p = tmp_path / "config.json"
    cfg.save(p)
    loaded = Config.load(p)
    assert loaded.keep_warm_s == 0
    assert loaded.wire_memory is False


def test_every_formatter_has_a_touch_hook():
    for f in (NoFormatter(), RulesFormatter(), OllamaFormatter("m", "http://localhost:1")):
        f.touch()   # no-op: must not raise and must not call out


def test_mlx_formatter_touch_without_a_loaded_model_is_a_noop():
    f = MLXFormatter("some/repo")
    f.touch()       # model not loaded: must not trigger a download or load
    assert f._model is None


def test_warm_due_survives_a_mistyped_interval():
    assert warm_due(0.0, 100.0, "45")            # hand-edited config: string still honoured
    assert not warm_due(0.0, 100.0, "nope")      # garbage disables quietly, no traceback


class _FakeModel:
    def __init__(self):
        self.touches = 0

    def touch(self):
        self.touches += 1


def _bare_pipeline():
    """The touch machinery without audio devices or model loads."""
    p = Pipeline.__new__(Pipeline)
    p._touch_lock = threading.Lock()
    p._touch_queued = False
    p._paste_spawn_lock = threading.Lock()
    p._repaste_inflight = False
    p._last_retarget_app = None
    p._jobs = queue.Queue()
    p._last_model_use = 0.0
    p.engine = _FakeModel()
    p.formatter = _FakeModel()
    return p


def test_queue_touch_dedupes():
    p = _bare_pipeline()
    p._queue_touch("a")
    p._queue_touch("b")
    assert p._jobs.qsize() == 1


def test_touch_skips_when_the_models_just_ran():
    p = _bare_pipeline()
    p._last_model_use = time.time()
    p._touch_models("test")
    assert p.engine.touches == 0     # a touch queued right behind a take must not re-run


def test_touch_runs_when_stale_and_stamps_the_use():
    p = _bare_pipeline()
    p._touch_models("test")
    assert p.engine.touches == 1 and p.formatter.touches == 1
    assert p._last_model_use > 0


def test_paste_block_never_enters_the_model_queue(monkeypatch):
    import dictdact.pipeline as pl

    p = _bare_pipeline()
    p._paste_lock = threading.Lock()
    p.cfg = Config()
    p.emit = lambda kind, payload=None: None
    pasted = threading.Event()
    monkeypatch.setattr(pl, "paste_text", lambda text, restore_clipboard=True: pasted.set() or True)

    class B:
        text = "hola"

    p.paste_block(B())
    assert pasted.wait(2)            # delivered by the fast lane…
    assert p._jobs.qsize() == 0      # …never queued behind model work
