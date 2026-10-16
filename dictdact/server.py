"""Localhost-only JSON API + static files for the DictDact window (and any browser on this Mac).

Bound to 127.0.0.1 on a random port; every request must carry the per-launch token
(`?t=` query or `X-DictDact-Token` header). Nothing is reachable from other hosts."""
from __future__ import annotations

import json
import logging
import math
import mimetypes
import secrets
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, autostart, perms
from .audio import MicMonitor, list_input_devices, resolve_input_device
from .config import CONFIG_PATH, DATA_DIR, HISTORY_PATH, Config
from .hotkey import parse_combo
from .models import missing_repos
from .stats import Stats

log = logging.getLogger("dictdact.server")
UI_DIR = Path(__file__).parent / "ui"

# keys the UI may change, with the allowed values (None = free text / bool)
EDITABLE = {
    "engine": ["whisper", "parakeet"],
    "formatter": ["mlx", "ollama", "rules", "off"],
    "formatter_style": ["clean", "improve", "translate_en", "translate_es", "summary"],
    "hotkey": ["ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_r", "cmd_r", "shift_r"],
    "retrieve_combo": None,     # free-form ("ctrl+alt+s"); parse_combo validates it below
    "language": [None, "es", "en"],
    "paste": [True, False],
    "paste_target": ["focus", "mouse"],
    "restore_clipboard": [True, False],
    "sounds": [True, False],
    "double_tap_lock": [True, False],
    "keep_recordings": [True, False],
    "voice_bank": [True, False],
    "dock_icon": [True, False],
    "show_window_at_launch": [True, False],
    "formatter_always": [True, False],
    "context_terms": [True, False],
    "microphone": None,
    "ollama_model": None,
    "max_gain_db": None,
}


def _db(x: float) -> float:
    return 20 * math.log10(x + 1e-9)


class Backend:
    """What the HTTP layer is allowed to see/do. Mutations that must run on the main thread
    are posted to ``app.events``; everything else is thread-safe enough for polling."""

    def __init__(self, app):
        self.app = app
        self.pipeline = app.pipeline
        self.cfg: Config = app.cfg
        self.stats = Stats(self.pipeline.history)
        self.monitor: MicMonitor | None = None
        self._monitor_until = 0.0
        self.pipeline.before_record = self.stop_monitor   # never hold the device twice

    # -- read ----------------------------------------------------------------------
    def state(self) -> dict:
        p = self.pipeline
        rec = p.recorder
        blocks, st = self.stats.load(time.strftime("%Y-%m-%d"))
        last = blocks[-1] if blocks else None
        return {
            "version": __version__,
            "ready": p.ready,
            "state": getattr(self.app, "_state", "idle"),
            "status": getattr(self.app, "_status_text", ""),
            "mic": {"name": rec.device.name if rec and rec.device else None,
                    "configured": self.cfg.microphone},
            "engine": p.engine.name if p.engine else self.cfg.engine,
            "formatter": p.formatter.name if p.formatter else self.cfg.formatter,
            "hotkey": self.cfg.hotkey,
            "perms": {"input_monitoring": perms.input_monitoring(), "accessibility": perms.accessibility(),
                      "microphone": perms.microphone()},
            "models_missing": missing_repos(self.cfg),
            "autostart": {"available": autostart.available(), "enabled": autostart.enabled() if autostart.available() else None},
            "last": asdict(last) if last else None,
            "counts": {"total": len(blocks), "today": st.today.takes,
                       "words_today": st.today.words,
                       "seconds_today": round(st.today.seconds),
                       "words_total": st.words, "seconds_total": round(st.seconds),
                       "words_per_day": round(st.words_per_day),
                       "days_used": st.days_used, "days_since_first": st.days_since_first,
                       "first_day": st.first_day},
            "paths": {"config": str(CONFIG_PATH), "history": str(HISTORY_PATH)},
        }

    def level(self) -> dict:
        rec = self.pipeline.recorder
        if rec and rec.recording:
            rms = rec.latest_rms()
            return {"source": "recording", "recording": True, "locked": bool(self.app.hotkeys.locked),
                    "rms": rms, "db": _db(rms), "peak_db": rec.level_dbfs(), "seconds": rec.seconds}
        if self.monitor and self.monitor.active:
            if time.time() > self._monitor_until:
                try:
                    self.stop_monitor()
                except Exception as e:
                    log.warning("monitor stop failed: %s", e)
                return {"source": "none", "recording": False, "rms": 0.0, "db": -120.0}
            rms, peak = self.monitor.level()
            return {"source": "monitor", "recording": False, "rms": rms, "db": _db(rms), "peak_db": _db(peak)}
        return {"source": "none", "recording": False, "rms": 0.0, "db": -120.0}

    def daily_stats(self, days: int = 0) -> dict:
        """Words per day for the chart. ``days`` trims to the last N calendar days (0 = all);
        the totals always cover the whole history, whatever window is on screen."""
        _, st = self.stats.load(time.strftime("%Y-%m-%d"))
        out = st.as_dict()
        if days > 0:
            out["days"] = out["days"][-days:]
        return out

    def history(self, q: str = "", limit: int = 300) -> dict:
        blocks = self.pipeline.history.all()[::-1]
        q = q.strip().lower()
        if q:
            blocks = [b for b in blocks if q in b.text.lower() or q in b.raw.lower()]
        return {"total": len(blocks), "items": [asdict(b) for b in blocks[:limit]]}

    def devices(self, refresh: bool = False) -> list[dict]:
        rec = self.pipeline.recorder
        if refresh and not (rec and rec.recording) and not (self.monitor and self.monitor.active):
            self.pipeline.run_on_audio_thread(lambda: list_input_devices(refresh=True), timeout=5)
        current = rec.device.name if rec and rec.device else None
        return [{"index": d.index, "name": d.name, "channels": d.channels, "samplerate": d.default_samplerate,
                 "default": d.is_default, "selected": d.name == current} for d in list_input_devices()]

    def config(self) -> dict:
        return asdict(self.cfg)

    # -- voice bank --------------------------------------------------------------------
    _export_job: dict = {}

    def voicebank(self) -> dict:
        from . import voicebank as vb

        texts = {blk.id: (blk.text or blk.raw) for blk in self.pipeline.history.last(60)}
        view = vb.window_view(on=bool(getattr(self.cfg, "voice_bank", False)), last=self.pipeline.voicebank.last,
                              target_min=float(getattr(self.cfg, "voice_bank_target_min", 180.0)), texts=texts,
                              pace_min_per_day=self._dictation_pace())
        view["export"] = dict(self._export_job)
        return view

    _pace: tuple = (0.0, {})

    def _dictation_pace(self) -> dict[str, float]:
        """Minutes dictated per day in each language over the last 7 days (cached a minute —
        the tab polls every few seconds)."""
        import time as _time
        from datetime import datetime, timedelta

        at, pace = self._pace
        if _time.time() - at < 60:
            return pace
        since = (datetime.now() - timedelta(days=7)).isoformat()
        totals: dict[str, float] = {}
        from .voicebank import infer_language

        for blk in self.pipeline.history.all():
            if blk.ts < since:
                continue
            lang = blk.language or infer_language(blk.raw or blk.text)   # Parakeet reports none
            if lang in ("es", "en"):
                totals[lang] = totals.get(lang, 0.0) + (blk.duration_s or 0.0)
        pace = {k: v / 60 / 7 for k, v in totals.items()}
        self._pace = (_time.time(), pace)
        return pace

    def voicebank_export(self, lang: str) -> dict:
        """Start an export on its own thread (hours of audio → MP3 takes a while); the tab polls
        ``export`` in /api/voicebank. The folder opens in Finder when it is done."""
        import subprocess
        import threading

        from . import voicebank as vb
        from .output import notify

        if lang not in vb.LANGS:
            raise ValueError(f"bad language: {lang!r}")
        if self._export_job.get("running"):
            raise ValueError("an export is already running")
        job_id = f"{lang}-{time.time():.3f}"      # every run is its own job: a repeated error is still news
        self._export_job = {"running": True, "lang": lang, "id": job_id}

        def run():
            try:
                r = vb.export(lang)
            except Exception as e:
                log.exception("voice bank export")
                self._export_job = {"running": False, "lang": lang, "id": job_id, "error": str(e)}
                notify("Voice bank", f"{vb.LANG_NAMES[lang]} export failed: {e}")
                return
            self._export_job = {"running": False, "lang": lang, "id": job_id, "result": r}
            subprocess.Popen(["open", r["dir"]])
            notify("Voice bank", f"{vb.LANG_NAMES[lang]}: {r['minutes']} min in {len(r['files'])} file(s) — folder opened")

        threading.Thread(target=run, name="dictdact-voice-export", daemon=True).start()
        return dict(self._export_job)

    # -- write --------------------------------------------------------------------------
    def set_config(self, changes: dict) -> dict:
        for key, value in changes.items():
            if key not in EDITABLE:
                raise ValueError(f"not editable: {key}")
            allowed = EDITABLE[key]
            if allowed is not None and value not in allowed:
                raise ValueError(f"bad value for {key}: {value!r}")
            if key == "max_gain_db":
                value = float(value)
            if key == "microphone":
                self.pipeline.set_microphone(value or None)       # saves
                if self.monitor and self.monitor.active:
                    self.start_monitor()                               # follow the new device
            elif key == "engine":
                self.pipeline.set_engine(value)                       # saves + reloads models
            elif key == "formatter":
                self.pipeline.set_formatter(value)
            elif key == "formatter_style":
                self.pipeline.set_formatter_style(value)            # saves + live prompt swap
            elif key in ("hotkey", "retrieve_combo"):
                if key == "retrieve_combo":
                    parse_combo(value)          # reject a typo before it reaches the listener
                setattr(self.cfg, key, value)
                self.cfg.save()
                self.app.events.put(("reload_hotkey", None))
            elif key == "language":
                self.pipeline.set_language(value)                     # saves + live on both models
            else:
                setattr(self.cfg, key, value)
                self.cfg.save()
                if key == "voice_bank" and not value:
                    self.pipeline.voicebank.switched_off()   # returns once nothing more can be saved
                if key == "dock_icon":
                    self.app.events.put(("dock_changed", None))
        self.app.events.put(("config_changed", None))
        return self.config()

    def start_monitor(self, seconds: float = 60.0) -> None:
        rec = self.pipeline.recorder
        if rec and rec.recording:
            return  # never open the device twice

        def _start():
            self._stop_monitor_here()
            device = resolve_input_device(self.cfg.microphone, refresh_if_missing=True)
            self.monitor = MicMonitor(device)
            self.monitor.start()
            self._monitor_until = time.time() + seconds

        self.pipeline.run_on_audio_thread(_start, timeout=8)

    def _stop_monitor_here(self) -> None:
        if self.monitor:
            m, self.monitor = self.monitor, None
            m.stop()

    def stop_monitor(self) -> None:
        if not self.monitor:
            return
        if threading.current_thread() is self.pipeline._audio_thread:
            self._stop_monitor_here()
        else:
            self.pipeline.run_on_audio_thread(self._stop_monitor_here, timeout=10)

    def vocab(self) -> dict:
        from .vocab import VOCAB_PATH

        from .vocab import vocab_lock

        with vocab_lock():
            v = self._vocab_in_sync()        # the file is the truth — a hand edit shows up on the next GET
        return {"terms": v.as_dicts(), "count": len(v), "path": str(VOCAB_PATH), "revision": v.revision}

    def _vocab_in_sync(self):
        """The file is the truth: a hand edit or a CLI write the app has not seen yet is adopted
        before any comparison or edit, so no editor can overwrite it."""
        from .vocab import Vocabulary

        v = self.pipeline.vocab
        if Vocabulary.load().revision != v.revision:
            log.info("vocabulary changed on disk — reloading before the edit")
            v = self.pipeline.set_vocabulary(None)
        return v

    def set_vocab(self, body: dict) -> dict:
        """{"terms": [...], "revision": h} replaces the list — only if ``revision`` (a content
        hash) is what the editor last saw, so a stale Dictionary tab can never wipe a CLI or
        hand edit; {"add": {...}} / {"remove": "term"} edit one entry; {"reload": true} re-reads
        the file. The whole operation holds the file lock: compare-and-swap, not read-then-write."""
        from .vocab import vocab_lock

        if body.get("reload"):
            self.pipeline.set_vocabulary(None)
            return self.vocab()
        with vocab_lock():
            v = self._vocab_in_sync()
            if "terms" in body:
                terms = body["terms"]
                if not isinstance(terms, list) or not all(isinstance(t, dict) and str(t.get("term", "")).strip() for t in terms):
                    raise ValueError("terms must be a list of {term, spoken[], note}")
                if "revision" not in body:
                    raise ValueError("a full replacement must carry the revision it is based on (GET /api/vocab)")
                if str(body["revision"]) != v.revision:
                    raise ValueError("The dictionary changed elsewhere (CLI or file) — reloaded, please redo the edit")
                self.pipeline.set_vocabulary(terms)
                return self.vocab()
            if "add" in body:
                a = body["add"]
                if not isinstance(a, dict) or not str(a.get("term", "")).strip():
                    raise ValueError("add needs a term")
                spoken = a.get("spoken") or []
                if isinstance(spoken, str):
                    spoken = [s for s in spoken.split(",")]
                v.add(str(a["term"]), [str(s) for s in spoken], str(a.get("note", "") or ""))
                self.pipeline.set_vocabulary(v.as_dicts())
                return self.vocab()
            if "remove" in body:
                if not v.remove(str(body["remove"])):
                    raise ValueError(f"not in the dictionary: {body['remove']}")
                self.pipeline.set_vocabulary(v.as_dicts())
                return self.vocab()
        raise ValueError("expected terms, add, remove or reload")

    def suggest_vocab(self, limit: int = 30) -> dict:
        """Names the speaker's own takes show Whisper is unsure about, clustered by sound."""
        from .learn import suggest, suggestions_as_dicts

        blocks = [b for b in self.pipeline.history.all() if b.raw]
        key = (len(blocks), self.pipeline.vocab.revision, limit)
        cached = getattr(self, "_suggest_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        raws = [b.raw for b in blocks]
        texts = [b.text for b in blocks if b.text]
        out = {"suggestions": suggestions_as_dicts(suggest(raws, self.pipeline.vocab, limit=limit, texts=texts)),
               "takes": len(raws)}
        self._suggest_cache = (key, out)
        return out

    def fix(self, block_id: str, text: str) -> dict:
        """The user corrected a dictation in the window: keep their version, and learn every
        name they respelled ("heard X → write Y") into the dictionary."""
        from .learn import apply_learned, learn_from_fix, record
        from .vocab import vocab_lock

        b = self.find(block_id)
        if b is None:
            raise ValueError("unknown dictation")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("the fixed text is empty — nothing saved")
        if len(text) > 50000:
            raise ValueError("the fixed text is too long")
        pairs = learn_from_fix(b.text, text)
        b.text = text
        self.pipeline.history.update(b)
        learned: list[dict] = []
        if pairs:
            with vocab_lock():
                v = self._vocab_in_sync()
                learned = apply_learned(v, pairs)
                self.pipeline.set_vocabulary(v.as_dicts())
            record(pairs, block_id)
            log.info("learned from a fix: %s", ", ".join(f"{h} → {w}" for h, w in pairs))
        return {"ok": True, "learned": learned, "block": asdict(b)}

    def find(self, block_id: str):
        return next((b for b in self.pipeline.history.all() if b.id == block_id), None)

    def paste(self, block_id: str) -> bool:
        b = self.find(block_id)
        if b is None:
            return False
        self.app.events.put(("paste_from_window", b))
        return True

    def copy(self, block_id: str) -> bool:
        from .output import set_clipboard_text

        b = self.find(block_id)
        if b is None:
            return False
        set_clipboard_text(b.text)
        return True

    def set_autostart(self, on: bool) -> bool:
        return autostart.set_enabled(on)


class Server:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.token = secrets.token_urlsafe(24)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="dictdact-http", daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    def start(self) -> None:
        self.thread.start()
        try:  # lets `dictdact ui` (and the owner) open the dashboard in a normal browser
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            p = DATA_DIR / "ui-url"
            p.write_text(self.url + "\n")
            p.chmod(0o600)
        except OSError:
            pass
        log.info("ui server on 127.0.0.1:%d", self.port)

    def stop(self) -> None:
        self.httpd.shutdown()

    def _handler_class(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            # The dashboard polls these ~20x a second for the mic meter while it is open.
            # Logging them rotated an hour of real diagnostics (paste outcomes, timings)
            # out of the file before anyone could read it.
            _QUIET = ("/api/level", "/api/state")

            def log_message(self, fmt, *args):  # keep the app log quiet
                if args and isinstance(args[0], str) and any(p in args[0] for p in self._QUIET):
                    return
                log.debug("http " + fmt, *args)

            # -- helpers ------------------------------------------------------------
            def _authorized(self, parsed) -> bool:
                q = parse_qs(parsed.query)
                tok = (q.get("t") or [None])[0] or self.headers.get("X-DictDact-Token")
                return tok == server.token

            def _json(self, obj, code=200):
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _static(self, name: str):
                path = (UI_DIR / name).resolve()
                if not str(path).startswith(str(UI_DIR.resolve())) or not path.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                data = path.read_bytes()
                ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or "javascript" in ctype else ""))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            # -- routes -------------------------------------------------------------------
            def do_GET(self):
                p = urlparse(self.path)
                if p.path.startswith("/api/") and not self._authorized(p):
                    self._json({"error": "unauthorized"}, 401)
                    return
                q = parse_qs(p.query)
                b = server.backend
                try:
                    if p.path in ("/", "/index.html"):
                        self._static("index.html")
                    elif p.path == "/api/state":
                        self._json(b.state())
                    elif p.path == "/api/level":
                        self._json(b.level())
                    elif p.path == "/api/stats":
                        self._json(b.daily_stats(int((q.get("days") or ["0"])[0])))
                    elif p.path == "/api/history":
                        self._json(b.history((q.get("q") or [""])[0], int((q.get("limit") or ["300"])[0])))
                    elif p.path == "/api/devices":
                        self._json(b.devices(refresh=(q.get("refresh") or ["0"])[0] == "1"))
                    elif p.path == "/api/config":
                        self._json(b.config())
                    elif p.path == "/api/vocab":
                        self._json(b.vocab())
                    elif p.path == "/api/voicebank":
                        self._json(b.voicebank())
                    elif p.path == "/api/vocab/suggest":
                        self._json(b.suggest_vocab(int((q.get("limit") or ["30"])[0])))
                    elif p.path.startswith("/api/"):
                        self._json({"error": "not found"}, 404)
                    else:
                        self._static(p.path.lstrip("/"))
                except Exception as e:
                    log.exception("GET %s failed", p.path)
                    self._json({"error": str(e)}, 500)

            def do_POST(self):
                p = urlparse(self.path)
                if not self._authorized(p):
                    self._json({"error": "unauthorized"}, 401)
                    return
                b = server.backend
                try:
                    body = self._body()
                    if p.path == "/api/config":
                        self._json(b.set_config(body))
                    elif p.path == "/api/vocab":
                        self._json(b.set_vocab(body))
                    elif p.path == "/api/voicebank/export":
                        self._json(b.voicebank_export(str(body.get("lang"))))
                    elif p.path == "/api/fix":
                        self._json(b.fix(str(body.get("id", "")), body.get("text", "")))
                    elif p.path == "/api/paste":
                        self._json({"ok": b.paste(body.get("id", ""))})
                    elif p.path == "/api/copy":
                        self._json({"ok": b.copy(body.get("id", ""))})
                    elif p.path == "/api/monitor":
                        if body.get("on"):
                            b.start_monitor(float(body.get("seconds", 60)))
                        else:
                            b.stop_monitor()
                        self._json({"ok": True, "active": bool(b.monitor and b.monitor.active)})
                    elif p.path == "/api/autostart":
                        self._json({"ok": b.set_autostart(bool(body.get("on")))})
                    elif p.path == "/api/open-settings":
                        import subprocess

                        pane = {"accessibility": "Privacy_Accessibility", "microphone": "Privacy_Microphone",
                                "login": None}.get(body.get("pane"))
                        url = ("x-apple.systempreferences:com.apple.LoginItems-Settings.extension" if body.get("pane") == "login"
                               else f"x-apple.systempreferences:com.apple.preference.security?{pane}")
                        subprocess.Popen(["open", url])
                        self._json({"ok": True})
                    elif p.path == "/api/reveal":
                        import subprocess

                        from .vocab import VOCAB_PATH

                        what = body.get("what")
                        if what == "voicebank":
                            from .voicebank import bank_dir

                            bank_dir().mkdir(parents=True, exist_ok=True)
                            subprocess.Popen(["open", str(bank_dir())])
                            self._json({"ok": True})
                            return
                        path = HISTORY_PATH if what == "history" else VOCAB_PATH if what == "vocab" else CONFIG_PATH
                        if what == "vocab" and not path.exists():
                            b.pipeline.vocab.save(path)
                        subprocess.Popen(["open", "-t", str(path)])
                        self._json({"ok": True})
                    else:
                        self._json({"error": "not found"}, 404)
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
                except Exception as e:
                    log.exception("POST %s failed", p.path)
                    self._json({"error": str(e)}, 500)

        return Handler
