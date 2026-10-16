import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

from dictdact.server import EDITABLE, Server


class FakeHistory:
    def __init__(self):
        self.blocks = []

    def all(self):
        return list(self.blocks)

    def latest(self):
        return self.blocks[-1] if self.blocks else None

    def update(self, block):
        self.blocks = [block if b.id == block.id else b for b in self.blocks]


def make_backend():
    from dictdact.server import Backend
    from dictdact.config import Config
    from dictdact.vocab import Term, Vocabulary
    cfg = Config()
    pipeline = SimpleNamespace(ready=True, recorder=SimpleNamespace(device=None, recording=False), engine=None,
                               formatter=None, history=FakeHistory(), set_microphone=lambda n: None,
                               set_engine=lambda e: setattr(cfg, "engine", e), set_formatter=lambda f: setattr(cfg, "formatter", f),
                               vocab=Vocabulary([Term("Acme Labs", ["ack me labs"])]))

    def set_language(v):
        cfg.language = v
        if pipeline.engine is not None:
            pipeline.engine.language = v

    def set_vocabulary(terms):
        # mirrors Pipeline.set_vocabulary without the worker thread
        if terms is None:
            pipeline.vocab = Vocabulary.load()
            return pipeline.vocab
        pipeline.vocab.replace_all(terms)
        pipeline.vocab.save()
        return pipeline.vocab

    pipeline.vocab.save()          # disk == memory at the start, as in the real app

    pipeline.set_language = set_language
    pipeline.set_vocabulary = set_vocabulary
    app = SimpleNamespace(pipeline=pipeline, cfg=cfg, hotkeys=SimpleNamespace(locked=False), _state="idle",
                          _status_text="Ready", events=SimpleNamespace(put=lambda e: None))
    b = Backend(app)
    b.cfg.save = lambda path=None: None
    return b


def get(url, token=None):
    req = urllib.request.Request(url, headers={"X-DictDact-Token": token} if token else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def post(url, token, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"X-DictDact-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_api_requires_token_and_serves_state():
    srv = Server(make_backend())
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        try:
            get(base + "/api/state")
            assert False, "unauthorized request succeeded"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        status, state = get(base + "/api/state", srv.token)
        assert status == 200 and state["ready"] is True and state["hotkey"] == "ctrl"
        status, lvl = get(base + "/api/level", srv.token)
        assert lvl["source"] == "none"
        # static UI is served without a token (nothing secret in it)
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            assert b"DictDact" in r.read()
        # config: only editable keys, validated
        status, cfg = post(base + "/api/config", srv.token, {"engine": "parakeet", "sounds": False})
        assert cfg["engine"] == "parakeet" and cfg["sounds"] is False
        try:
            post(base + "/api/config", srv.token, {"whisper_model": "evil"})
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 400
        try:
            post(base + "/api/config", srv.token, {"engine": "nope"})
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert srv.httpd.server_address[0] == "127.0.0.1"
        # history: independent total + limit
        from dictdact.history import Block
        for i in range(5):
            srv.backend.pipeline.history.blocks.append(Block.new(text=f"t{i}", raw=f"r{i}"))
        status, h = get(base + "/api/history?limit=2", srv.token)
        assert h["total"] == 5 and len(h["items"]) == 2 and h["items"][0]["text"] == "t4"
        status, h = get(base + "/api/history?q=t3", srv.token)
        assert h["total"] == 1 and h["items"][0]["text"] == "t3"
        # copy works, unknown id is not ok
        bid = srv.backend.pipeline.history.blocks[-1].id
        import dictdact.server as S
        S.set_clipboard_text = lambda t: None
        status, r = post(base + "/api/copy", srv.token, {"id": "nope"})
        assert r["ok"] is False
        # language applies live to the loaded engine
        eng = SimpleNamespace(name="whisper", language=None)
        srv.backend.pipeline.engine = eng
        post(base + "/api/config", srv.token, {"language": "es"})
        assert eng.language == "es" and srv.backend.cfg.language == "es"
        status, devs = get(base + "/api/devices", srv.token)
        assert isinstance(devs, list)
        # vocabulary: read, add, remove, replace — always the full list back
        status, v = get(base + "/api/vocab", srv.token)
        assert v["count"] == 1 and v["terms"][0]["term"] == "Acme Labs" and v["path"].endswith("vocabulary.json")
        status, v = post(base + "/api/vocab", srv.token, {"add": {"term": "Hyrox", "spoken": "hirox, hierox", "note": "race"}})
        assert v["count"] == 2 and v["terms"][1] == {"term": "Hyrox", "spoken": ["hirox", "hierox"], "note": "race", "fuzzy": False, "learned": []}
        assert srv.backend.pipeline.vocab.apply("hierox and ack me labs") == "Hyrox and Acme Labs"
        status, v = post(base + "/api/vocab", srv.token, {"remove": "hyrox"})
        assert v["count"] == 1
        try:
            post(base + "/api/vocab", srv.token, {"remove": "nope"})
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 400
        try:
            post(base + "/api/vocab", srv.token, {"terms": [{"spoken": ["x"]}]})
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 400
        try:
            post(base + "/api/vocab", srv.token, {"terms": [{"term": "Acme", "spoken": [], "note": ""}]})
            assert False, "a full replacement without a revision was accepted"
        except urllib.error.HTTPError as e:
            assert e.code == 400
        status, v = post(base + "/api/vocab", srv.token, {"terms": [{"term": "Acme", "spoken": [], "note": ""}],
                                                          "revision": v["revision"]})
        assert [t["term"] for t in v["terms"]] == ["Acme"]
        # a stale editor (older revision) cannot wipe a change made elsewhere
        rev = v["revision"]
        srv.backend.pipeline.vocab.add("FromCLI", ["from cli"])          # the CLI edited meanwhile…
        srv.backend.pipeline.vocab.save()                                  # …and wrote the file, as it does
        try:
            post(base + "/api/vocab", srv.token, {"terms": [{"term": "Acme"}], "revision": rev})
            assert False, "stale save accepted"
        except urllib.error.HTTPError as e:
            assert e.code == 400
        status, v = get(base + "/api/vocab", srv.token)
        assert [t["term"] for t in v["terms"]] == ["Acme", "FromCLI"] and v["revision"] != rev
        status, v = post(base + "/api/vocab", srv.token, {"terms": [{"term": "Acme"}], "revision": v["revision"]})
        assert [t["term"] for t in v["terms"]] == ["Acme"]
        # a HAND edit of the file the app has not seen yet is adopted, not overwritten
        from dictdact.vocab import Term, Vocabulary
        stale_rev = v["revision"]
        Vocabulary([Term("Acme"), Term("HandEdit", ["hand edit"])]).save()
        try:
            post(base + "/api/vocab", srv.token, {"terms": [{"term": "Acme"}], "revision": stale_rev})
            assert False, "a stale tab overwrote a hand edit"
        except urllib.error.HTTPError as e:
            assert e.code == 400
        status, v = get(base + "/api/vocab", srv.token)
        assert [t["term"] for t in v["terms"]] == ["Acme", "HandEdit"]
        # and a plain GET after another hand edit already shows the file, not stale memory
        Vocabulary([Term("Acme"), Term("HandEdit", ["hand edit"]), Term("DiskNew")]).save()
        status, v = get(base + "/api/vocab", srv.token)
        assert [t["term"] for t in v["terms"]] == ["Acme", "HandEdit", "DiskNew"]
        Vocabulary([Term("Acme"), Term("HandEdit", ["hand edit"])]).save()
        # and a single-entry edit lands on top of the hand edit, never instead of it
        status, v = post(base + "/api/vocab", srv.token, {"add": {"term": "Later"}})
        assert [t["term"] for t in v["terms"]] == ["Acme", "HandEdit", "Later"]
        try:
            get(base + "/api/vocab")
            assert False, "vocab served without a token"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        # learning: suggestions from history, and a fix that teaches the dictionary
        hist = srv.backend.pipeline.history
        for raw in ["fuimos a Hyrox hoy", "otra vez a Hirox", "y luego Hierox otra vez", "Hyrox de nuevo",
                    "mañana Bevel y ManyChat", "el plan de Bebel", "vamos con Bevel otra vez", "hoy Bevel siempre",
                    "revisa lo de Bebel también"]:
            hist.blocks.append(Block.new(text=raw, raw=raw))
        status, sg = get(base + "/api/vocab/suggest?limit=10", srv.token)
        terms = {x["term"]: x for x in sg["suggestions"]}
        assert sg["takes"] >= 8 and "Hyrox" in terms and "Hirox" in terms["Hyrox"]["spoken"] and terms["Hyrox"]["strong"]
        assert "Bevel" in terms and terms["Bevel"]["spoken"] == ["Bebel"]
        blk = Block.new(text="fuimos a Hirox con Manichat y John", raw="fuimos a Hirox con Manichat y John")
        hist.blocks.append(blk)
        status, fx = post(base + "/api/fix", srv.token, {"id": blk.id, "text": "fuimos a Hyrox con ManyChat y Daniel"})
        assert fx["ok"] and [(l["heard"], l["term"]) for l in fx["learned"]] == [("Hirox", "Hyrox"), ("Manichat", "ManyChat")]
        assert fx["block"]["text"] == "fuimos a Hyrox con ManyChat y Daniel"
        assert hist.blocks[-1].text == "fuimos a Hyrox con ManyChat y Daniel"
        # the learned spellings apply exactly (a fix never arms sound-alike matching)
        assert srv.backend.pipeline.vocab.apply("vamos a Hirox con Manichat") == "vamos a Hyrox con ManyChat"
        assert srv.backend.pipeline.vocab.apply("vamos a Hierox") == "vamos a Hierox"
        from dictdact.learn import LEARNED_PATH
        assert LEARNED_PATH.exists() and "Manichat" in LEARNED_PATH.read_text(encoding="utf-8")
        status, sg = get(base + "/api/vocab/suggest", srv.token)
        assert "Hyrox" not in {x["term"] for x in sg["suggestions"]}      # now a term → no longer suggested
        try:
            post(base + "/api/fix", srv.token, {"id": "nope", "text": "x"})
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.stop()


def test_stats_serves_words_per_day_and_the_sidebar_totals():
    from dictdact.history import Block
    srv = Server(make_backend())
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        hist = srv.backend.pipeline.history
        hist.blocks.append(Block(id="a", ts="2026-09-01T10:00:00", text="three words here", raw="", duration_s=30.0))
        hist.blocks.append(Block.new(text="said today", raw=""))
        status, st = get(base + "/api/stats", srv.token)
        assert status == 200
        assert st["days"][0] == {"date": "2026-09-01", "takes": 1, "words": 3, "seconds": 30.0}
        assert st["days"][-1]["date"] == st["today"]["date"]      # zero-filled right up to today
        assert st["today"]["words"] == 2 and st["totals"]["words"] == 5
        assert st["totals"]["first_day"] == "2026-09-01" and st["totals"]["best"]["words"] == 3
        # a shorter window trims the bars, never the totals
        status, w = get(base + "/api/stats?days=2", srv.token)
        assert len(w["days"]) == 2 and w["totals"]["words"] == 5
        # the sidebar reads the same numbers off /api/state
        status, state = get(base + "/api/state", srv.token)
        c = state["counts"]
        assert c["words_total"] == 5 and c["words_today"] == 2 and c["today"] == 1
        assert c["first_day"] == "2026-09-01" and c["days_used"] == 2
        assert c["words_per_day"] == round(5 / c["days_since_first"])
        assert state["last"]["text"] == "said today"              # the cached read still knows the last take
        try:
            get(base + "/api/stats")
            assert False, "stats served without a token"
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        srv.stop()


def test_editable_keys_are_a_closed_set():
    assert "whisper_model" not in EDITABLE and "mlx_formatter_model" not in EDITABLE


def test_fix_refuses_empty_text():
    srv = Server(make_backend())
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        from dictdact.history import Block
        blk = Block.new(text="hola mundo", raw="hola mundo")
        srv.backend.pipeline.history.blocks.append(blk)
        try:
            post(base + "/api/fix", srv.token, {"id": blk.id, "text": "   "})
            assert False, "an empty fix was accepted"
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert srv.backend.pipeline.history.blocks[-1].text == "hola mundo"
    finally:
        srv.stop()
