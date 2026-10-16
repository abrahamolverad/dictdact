"""Record → preprocess → transcribe → structure → paste, on a single worker thread."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import TARGET_SR, Recorder, analyze_and_normalize, resolve_input_device, save_wav, to_target_sr
from .config import RECORDINGS_DIR, Config
from .context import ContextReader
from .format import make_formatter
from .history import Block, History
from .output import frontmost_app_name, notify, paste_text, play, set_clipboard_text
from . import target
from .transcribe import SUPPORTED as LANGUAGES, collapse_repeats, make_engine
from .vocab import Vocabulary
from .voicebank import VoiceBank
from .warmth import HOTKEY_PREWARM_IDLE_S, warm_due

log = logging.getLogger("dictdact.pipeline")

Event = tuple[str, object]


class Pipeline:
    def __init__(self, cfg: Config, emit: Callable[[str, object], None] | None = None):
        self.cfg = cfg
        self.history = History()
        self.emit = emit or (lambda kind, payload=None: None)
        self.vocab = Vocabulary.load()       # the user's dictionary: prompt tail + glossary + replacements
        self._protect_common_words(self.vocab)
        self.context = ContextReader(getattr(cfg, "context_terms", True))   # names visible in the active window
        # raw takes kept for voice cloning (cfg.voice_bank); milestones reach the speaker as notifications
        self.voicebank = VoiceBank(notify=lambda title, msg: (notify(title, msg), self.emit("status", msg)),
                                   enabled=lambda: bool(getattr(self.cfg, "voice_bank", False)))
        self.voicebank.on_change = lambda: self.emit("voicebank", None)    # the menu redraws its progress
        self.engine = None
        self.formatter = None
        self.ready = False
        self.last_repaste_app: str | None = None
        self.recorder: Recorder | None = None
        self._app_at_start: str | None = None
        self.before_record = None   # hook: e.g. the dashboard stops its mic-test monitor
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="dictdact-worker", daemon=True)
        self._worker.start()
        # Every PortAudio/CoreAudio call runs on this thread, never on the hotkey (event-tap)
        # thread: a hung CoreAudio stop used to freeze key delivery for the whole machine.
        self._audio_jobs: queue.Queue = queue.Queue()
        self._audio_thread = threading.Thread(target=self._run_audio, name="dictdact-audio", daemon=True)
        self._audio_thread.start()
        self.audio_hung = False
        # Keep the model weights resident under memory pressure (see dictdact/warmth.py).
        self._last_model_use = 0.0
        self._touch_lock = threading.Lock()
        self._touch_queued = False
        self._paste_lock = threading.Lock()   # clipboard/⌘V is shared by the worker and the paste lane
        self._paste_spawn_lock = threading.Lock()
        self._repaste_inflight = False        # macOS key-repeat fires Ctrl+Alt+V many times; one lane at a time
        self._last_retarget_app: str | None = None
        self._warm_thread = threading.Thread(target=self._warm_loop, name="dictdact-keepwarm", daemon=True)
        self._warm_thread.start()
        self.set_microphone(cfg.microphone, save=False)

    # -- worker ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception as e:
                log.exception("job failed")
                play("error", self.cfg.sounds)
                self.emit("error", str(e))
            finally:
                self._jobs.task_done()

    def _run_audio(self) -> None:
        while True:
            job = self._audio_jobs.get()
            try:
                job()
            except Exception as e:
                log.exception("audio job failed")
                play("error", self.cfg.sounds)
                self.emit("error", str(e))
            finally:
                self._audio_jobs.task_done()

    def run_on_audio_thread(self, fn, timeout: float | None = 10.0):
        """Run ``fn`` on the audio thread and wait for its result (used by the dashboard)."""
        done = threading.Event()
        box: dict = {}

        def job():
            try:
                box["result"] = fn()
            except Exception as e:
                box["error"] = e
            finally:
                done.set()

        self._audio_jobs.put(job)
        if not done.wait(timeout):
            raise TimeoutError("audio device is not responding")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def load_models(self) -> None:
        self._jobs.put(self._load_models)

    def _load_models(self) -> None:
        self.ready = False
        self._set_wired_limit(8 << 30, "before load")   # buffers allocated from here on may be wired
        self.emit("status", "Loading speech model…")
        self.engine = make_engine(self.cfg, self.vocab.whisper_terms())
        self.engine.warmup()
        self.emit("status", "Loading formatter…")
        self.formatter = make_formatter(self.cfg, self.vocab)
        self.formatter.warmup()
        self._wire_loaded_models()
        self._last_model_use = time.time()
        self.ready = True
        self.emit("ready", f"{self.engine.name} + {self.formatter.name}")
        log.info("ready: engine=%s formatter=%s mic=%s language=%s vocabulary=%d terms",
                 self.engine.name, self.formatter.name,
                 self.recorder.device.name if self.recorder and self.recorder.device else "default",
                 self.cfg.language or "auto", len(self.vocab))

    def wait_ready(self, timeout: float | None = None) -> bool:
        t0 = time.time()
        while not self.ready:
            if timeout is not None and time.time() - t0 > timeout:
                return False
            time.sleep(0.1)
        return True

    # -- staying fast under memory pressure -----------------------------------
    def _set_wired_limit(self, limit_bytes: int, note: str) -> None:
        """Wire MLX's buffer cache so the weights can never be paged out. Without this, a
        memory-hungry neighbour (a 100 GB ComfyUI job…) evicts the idle weights and the next
        dictation repays the page-in after the key is released."""
        if not self.cfg.wire_memory:
            return
        try:
            import mlx.core as mx

            m = getattr(mx, "metal", None)
            set_limit = getattr(mx, "set_wired_limit", None) or getattr(m, "set_wired_limit", None)
            if set_limit is None:
                log.info("this mlx has no set_wired_limit — relying on keep-warm touches only")
                return
            info_fn = getattr(mx, "device_info", None) or getattr(m, "device_info", None)
            if info_fn is not None:
                # An 8 GB Mac rejects a limit above its recommended working set outright —
                # cap instead of losing wiring entirely there.
                rec = int(info_fn().get("max_recommended_working_set_size", 0) or 0)
                if rec:
                    limit_bytes = min(int(limit_bytes), int(rec * 0.8))
            set_limit(int(limit_bytes))
            log.info("wired limit %s: %.1f GB", note, limit_bytes / 2**30)
        except Exception as e:
            log.warning("could not set the wired limit (%s): %s", note, e)

    def _wire_loaded_models(self) -> None:
        if not self.cfg.wire_memory:
            return
        try:
            import mlx.core as mx

            m = getattr(mx, "metal", None)
            get_active = getattr(mx, "get_active_memory", None) or getattr(m, "get_active_memory", None)
            if get_active is None:
                return
            # Warmup leaves GB-sized transients in MLX's buffer cache; sizing from active
            # memory alone would set the limit below what is really resident and the shrink
            # could silently unwire the weights themselves. Drop the transients, then wire
            # what is actually still held (weights + prompt cache) plus headroom.
            clear = getattr(mx, "clear_cache", None) or getattr(m, "clear_cache", None)
            if clear is not None:
                clear()
            get_cache = getattr(mx, "get_cache_memory", None) or getattr(m, "get_cache_memory", None)
            need = int(get_active()) + int(get_cache() if get_cache else 0) + (768 << 20)
            self._set_wired_limit(need, "for the loaded models")
        except Exception as e:
            log.warning("could not size the wired limit: %s", e)

    def _queue_touch(self, reason: str) -> None:
        with self._touch_lock:
            if self._touch_queued:
                return
            self._touch_queued = True
        self._jobs.put(lambda: self._touch_models(reason))

    def _touch_models(self, reason: str) -> None:
        try:
            now = time.time()
            if not warm_due(self._last_model_use, now, 3.0):
                log.debug("%s touch skipped — models ran %.1fs ago", reason, now - self._last_model_use)
                return
            t0 = time.time()
            # A touch decodes pure zeros; the engines' hallucination-filter chatter would land
            # in the log every keep_warm_s and rotate real evidence out. These loggers are only
            # ever used from this worker thread, so the mute cannot hide another take's lines.
            quiet = [logging.getLogger("dictdact.stt"), logging.getLogger("dictdact.format")]
            levels = [lg.level for lg in quiet]
            for lg in quiet:
                lg.setLevel(logging.WARNING)
            try:
                if self.engine is not None:
                    self.engine.touch()
                if self.formatter is not None:
                    self.formatter.touch()
            finally:
                for lg, lv in zip(quiet, levels):
                    lg.setLevel(lv)
            dt = time.time() - t0
            self._last_model_use = time.time()
            if dt > 1.0:
                log.info("models were cold or the GPU busy — %s touch took %.1fs (pressure from a neighbouring app)",
                         reason, dt)
            else:
                log.debug("%s touch %.2fs", reason, dt)
        except Exception:
            log.exception("keep-warm touch failed")
        finally:
            with self._touch_lock:
                self._touch_queued = False

    def _warm_loop(self) -> None:
        while True:
            time.sleep(5.0)
            try:
                if not self.ready or (self.recorder and self.recorder.recording):
                    continue
                if not self._jobs.empty():
                    continue
                if warm_due(self._last_model_use, time.time(), self.cfg.keep_warm_s):
                    self._queue_touch("heartbeat")
            except Exception:
                log.exception("keep-warm loop failed")

    # -- configuration --------------------------------------------------------
    def set_microphone(self, name: str | None, save: bool = True) -> None:
        if self.recorder and self.recorder.recording:
            self.recorder.cancel()
        device = resolve_input_device(name)
        self.recorder = Recorder(device, self.cfg.max_seconds, preferred_name=name)
        self.cfg.microphone = name
        if save:
            self.cfg.save()
        log.info("microphone: %s", device.name if device else "system default")

    def set_engine(self, engine: str) -> None:
        self.cfg.engine = engine
        self.cfg.save()
        self.load_models()

    def set_formatter(self, formatter: str) -> None:
        self.cfg.formatter = formatter
        self.cfg.save()
        self.load_models()

    def set_formatter_style(self, style: str) -> None:
        """Swap the result style without reloading the model: update the system prompt and
        rebuild the KV prefix cache, on the worker thread (all MLX runs there)."""
        self.cfg.formatter_style = style
        self.cfg.save()

        def apply():
            f = self.formatter
            if f is None:
                return
            f.style = style
            self._rebuild_prompt_cache()
            log.info("result style: %s", style)

        self._jobs.put(apply)

    def _rebuild_prompt_cache(self) -> None:
        """The system prompt changed (style, language lock or glossary): recompute the KV
        prefix cache. Worker thread only — all MLX runs there."""
        f = self.formatter
        if f is not None and getattr(f, "_model", None) is not None and hasattr(f, "_build_prefix_cache"):
            f._build_prefix_cache()
        self._last_model_use = time.time()

    def set_language(self, language: str | None) -> None:
        """Lock the dictation language (es/en) or go back to auto (None). Both models follow:
        Whisper forces the language token + a monolingual prompt, the formatter is told to
        write in that language. Applied on the worker, live — no reload."""
        language = language if language in LANGUAGES else None
        self.cfg.language = language
        self.cfg.save()

        def apply():
            if self.engine is not None:
                self.engine.language = language
            if self.formatter is not None:
                self.formatter.language = language
                self._rebuild_prompt_cache()
            log.info("language: %s", language or "auto")
            msg = {"es": "Language: Español (locked)", "en": "Language: English (locked)"}.get(
                language, "Language: auto (Spanish + English)")
            if language and self.engine is not None and self.engine.name == "parakeet":
                # Parakeet has no language token: the lock can only steer the formatter.
                log.warning("language lock with Parakeet applies to the formatter only (no language token)")
                msg += " — Parakeet has no language token; switch to Whisper for a real lock"
            self.emit("status", msg)

        self._jobs.put(apply)

    def _protect_common_words(self, vocab: Vocabulary) -> None:
        """Sound-alike matching must never rewrite a word the speaker also uses as a plain
        word ("cloud", "super", "day"): learned from their own history, once per load."""
        try:
            from .learn import protected_words

            vocab.protected = protected_words([b.raw for b in self.history.all() if b.raw])
        except Exception:
            log.exception("could not build the protected-word list")

    def set_vocabulary(self, terms: list[dict] | None = None) -> Vocabulary:
        """Replace the dictionary (``terms`` = the full list as dicts) or reload it from disk
        (None — the CLI or a hand edit changed the file). Whisper's prompt tail, the LLM
        glossary and the replacement rules all follow, live."""
        if terms is None:
            self.vocab = Vocabulary.load()
            self._protect_common_words(self.vocab)
        else:
            self.vocab.replace_all(terms)
            self.vocab.save()
        vocab = self.vocab

        def apply():
            if self.engine is not None:
                self.engine.terms = vocab.whisper_terms()
            if self.formatter is not None:
                self.formatter.vocab = vocab
                self._rebuild_prompt_cache()
            log.info("vocabulary: %d terms", len(vocab))

        self._jobs.put(apply)
        return vocab

    # -- recording (called from the hotkey thread: only enqueue, never touch audio here) ------
    def start_recording(self) -> None:
        if not self.ready:
            self.emit("status", "Still loading models — try again in a moment")
            play("error", self.cfg.sounds)
            return
        self._app_at_start = frontmost_app_name()
        self._audio_jobs.put(self._do_start)

    def cancel_recording(self) -> None:
        self._audio_jobs.put(self._do_cancel)

    def stop_recording(self) -> None:
        app = self._app_at_start
        self._audio_jobs.put(lambda: self._do_stop(app))

    # -- audio-thread side -----------------------------------------------------------------
    def _do_start(self) -> None:
        if self.recorder.recording:
            return
        if self.before_record:
            try:
                self.before_record()
            except Exception:
                log.exception("before_record hook failed")
        try:
            # The voice bank needs the take at the mic's own rate; otherwise CoreAudio hands us
            # the 16 kHz the ASR wants, exactly as before.
            # A take belongs to the bank's state when it STARTED: switching the bank off during
            # this take (even back on before it is transcribed) keeps it out of the bank. The
            # generation is read BEFORE the setting: 'off' flips the setting, then bumps the
            # generation, so an 'off' landing between the two reads still leaves a stale stamp.
            self._take_generation = self.voicebank.generation
            self.recorder.native = bool(getattr(self.cfg, "voice_bank", False))
            self._take_banked = self.recorder.native      # opted in when the take started (the
                                                          # sample rate alone proves nothing: a mic
                                                          # that rejects 16 kHz opens native anyway)
            self.recorder.start()
        except Exception as e:
            log.error("could not open microphone: %s", e)
            play("error", self.cfg.sounds)
            self.emit("error", f"Microphone error: {e}")
            return
        dev = self.recorder.device
        if dev is not None and dev.fallback_reason:
            self.emit("status", f"Using {dev.name} ({dev.fallback_reason})")
        # Only chirp once this is clearly a hold, not a Ctrl+<key> shortcut passing through.
        threading.Timer(self.cfg.hold_threshold_ms / 1000.0, self._hold_confirmed).start()
        self.emit("recording", None)

    def _hold_confirmed(self) -> None:
        """hold_threshold_ms after the mic opened: past this point it is (almost always) a
        dictation, not a Ctrl+<key> shortcut passing through."""
        if not self.recorder.recording:
            return
        play("start", self.cfg.sounds)
        # Names visible in the app being dictated into → this take's Whisper prompt. Read on
        # its own thread with hard bounds; the take never waits for it.
        self.context.enabled = getattr(self.cfg, "context_terms", True)
        self.context.start(exclude=set(self.vocab.whisper_terms()))
        # Re-warm while the user speaks: an evicted-weights page-in overlaps the recording
        # instead of delaying the transcription. A slow combo can still slip past this gate,
        # which is fine — re-pastes run on their own lane and never wait behind a touch.
        if warm_due(self._last_model_use, time.time(), HOTKEY_PREWARM_IDLE_S):
            self._queue_touch("hotkey")

    def _do_cancel(self) -> None:
        if self.recorder.recording:
            _audio, _sr, stream = self.recorder.take_raw()   # silent: usually a Ctrl+<key> shortcut
            self._close(stream)
        self.emit("idle", None)

    def _do_stop(self, app: str | None) -> None:
        if not self.recorder.recording:
            return
        audio, sr, stream = self.recorder.take_raw()        # the words are safe from here on
        device = self.recorder.device
        play("stop", self.cfg.sounds)
        self.emit("processing", None)
        # transcribe + paste regardless of the close; resampling to 16 kHz happens on the worker
        generation = getattr(self, "_take_generation", None)
        banked = getattr(self, "_take_banked", False)
        self._jobs.put(lambda: self._process(audio, app, sr=sr, device=device, generation=generation,
                                             bank=banked))
        self._close(stream)

    def _close(self, stream) -> None:
        if self.recorder.close_stream(stream):
            return
        # CoreAudio hung while stopping the stream. The take is already queued; once it is
        # pasted, restart the app (launchd brings it back in ~5 s) rather than stay half-dead.
        self.audio_hung = True
        self.emit("error", "Audio device hung — DictDact will restart itself after this take")

        def _restart_after_take():
            import os
            import sys

            log.error("restarting after audio device hang")
            for h in logging.getLogger("dictdact").handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            os._exit(1)

        self._jobs.put(_restart_after_take)

    # -- processing -------------------------------------------------------------
    def _process(self, audio: np.ndarray, app: str | None, sr: int = TARGET_SR, device=None,
                 generation: int | None = None, bank: bool = True) -> Block | None:
        timings: dict[str, float] = {}
        self._last_model_use = time.time()   # the engine is about to run: hold the heartbeat off
        t0 = time.time()
        native = audio if sr != TARGET_SR else None      # the voice bank's copy, untouched
        audio = to_target_sr(audio, sr)
        x, stats = analyze_and_normalize(audio, max_gain_db=self.cfg.max_gain_db)
        timings["preprocess"] = round(time.time() - t0, 3)
        log.info("audio: %s", stats.summary())
        if not stats.has_speech:
            if stats.peak_dbfs <= -150 and stats.duration_s >= 0.5:
                from . import perms

                mic = self.recorder.device.name if self.recorder.device else "default mic"
                msg = (f"{mic} delivered pure silence (mic permission = {perms.microphone()}). "
                       "Allow DictDact in System Settings → Privacy & Security → Microphone, or pick another mic.")
                log.error(msg)
                self.emit("error", msg)
                return None
            self.emit("idle", "No speech detected")
            return None
        audio_path = None
        if self.cfg.keep_recordings:
            audio_path = RECORDINGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
            save_wav(audio_path, x)

        t1 = time.time()
        ctx_app, ctx_terms = self.context.take()
        if ctx_terms:
            log.info("context: %d terms from %s", len(ctx_terms), ctx_app or "?")   # names stay off the log
        transcript = self.engine.transcribe(x, extra_terms=ctx_terms)
        timings["transcribe"] = round(time.time() - t1, 3)
        log.info("raw (%s, %.1fs): %s", transcript.language, transcript.elapsed_s, transcript.text)
        if not transcript.text:
            # Keep the audio so an empty result can be diagnosed later (small: 16 kHz mono).
            failed = RECORDINGS_DIR / f"failed-{time.strftime('%Y%m%d-%H%M%S')}.wav"
            try:
                save_wav(failed, x)
                log.warning("nothing recognized in %.1fs of %s-level speech — audio kept at %s", stats.duration_s, stats.mode, failed)
            except OSError:
                pass
            hint = (f"Nothing recognized — input very low ({stats.speech_dbfs:.0f} dB): get closer or raise the mic gain"
                    if stats.mode == "whisper" else "Nothing recognized")
            self.emit("idle", hint)
            return None

        t2 = time.time()
        self.emit("status", "Structuring…")
        text = self.formatter.format(transcript.text)
        # Delivery invariant: a repetition loop never reaches the paste, whichever engine or
        # formatter path produced it (the LLM can loop too, inside the length guard).
        text = collapse_repeats(text)
        timings["format"] = round(time.time() - t2, 3)
        self._last_model_use = time.time()
        log.info("text: %s", text)
        log.info("timings: preprocess=%.2fs transcribe=%.2fs format=%.2fs",
                 timings["preprocess"], timings["transcribe"], timings["format"])

        block = Block.new(
            text=text, raw=transcript.text, language=transcript.language, engine=transcript.engine,
            formatter=self.formatter.name, mode=stats.mode, duration_s=round(stats.duration_s, 2),
            app=app, audio_path=str(audio_path) if audio_path else None, timings=timings,
        )
        fallback = getattr(self.formatter, "last_fallback", None)
        if fallback:
            block.formatter = f"{self.formatter.name}→rules"
            self.emit("status", f"{fallback} — kept your words with the deterministic cleanup")
        self.history.append(block)  # saved BEFORE pasting, so it can never be lost
        block.pasted = self._deliver(block.text)
        if block.pasted:
            if self._last_retarget_app:
                block.app = self._last_retarget_app   # the app that actually received the text
            self.history.update(block)
        play("done", self.cfg.sounds)
        self.emit("done", block)
        if bank and native is not None and getattr(self.cfg, "voice_bank", False):
            # After the paste, on the bank's own thread: banking never delays a dictation.
            self.voicebank.offer(
                native, sr, block_id=block.id, ts=block.ts, language=transcript.language or self.cfg.language,
                text=block.raw,
                mode=stats.mode, device=device.name if device else None,
                fallback=device.fallback_reason if device else None, preferred=self.cfg.microphone,
                target_min=float(getattr(self.cfg, "voice_bank_target_min", 180.0)),
                **({"generation": generation} if generation is not None else {}))
        return block

    def _deliver(self, text: str) -> bool:
        with self._paste_lock:
            if self.cfg.paste:
                if not self._mouse_retarget_ok():
                    set_clipboard_text(text)      # secure field under the pointer: never ⌘V
                    log.info("delivery: clipboard only (the ⌘V was refused)")
                    return False
                ok = paste_text(text, restore_clipboard=self.cfg.restore_clipboard)
                # Every take says where its words went. Without this line a paste that goes
                # nowhere leaves no trace at all, and "it didn't paste" cannot be diagnosed
                # afterwards — which is exactly what happened.
                log.info("delivery: %s → %s", "⌘V sent" if ok else "clipboard only",
                         self._last_retarget_app or frontmost_app_name() or "?")
                return ok
            if self.cfg.copy_to_clipboard:
                set_clipboard_text(text)
                log.info("delivery: clipboard (pasting is off)")
            return False

    def _mouse_retarget_ok(self) -> bool:
        """paste_target='mouse': front the app under the pointer before the ⌘V. True = go
        ahead and paste; False = a secure (password) field would receive it, clipboard
        only. Every failure path degrades to the classic focus behavior — text is never
        lost. The secure check runs on EVERY mouse-mode paste, including when no app
        switch is needed: pointing at the password box you are focused on is the common
        case, not the exotic one."""
        if getattr(self.cfg, "paste_target", "focus") != "mouse":
            return True
        self._last_retarget_app = None
        try:
            pt = target.mouse_point()
            windows = target.onscreen_windows()
            tgt = target.pick_paste_target(pt, windows, os.getpid())
            if tgt is not None:
                # CGWindowList cannot see hit-testability: a click-through transparent
                # overlay or a save-panel XPC sheet wins the geometric pick. Prefer what
                # the system's own AX hit-test reaches when it maps to a real window.
                ax_pid = target.ax_pid_at_point(*pt)
                if ax_pid and ax_pid not in (tgt.pid, os.getpid()):
                    ax_win = next((w for w in windows if w.pid == ax_pid and w.layer == 0 and w.contains(*pt)), None)
                    if ax_win is not None:
                        log.debug("AX hit-test overrides the pick: %s → %s", tgt.app, ax_win.app)
                        tgt = ax_win
            front = target.frontmost_pid()
            if tgt is None or (front is not None and tgt.pid == front):
                check = front if tgt is None else tgt.pid     # no switch needed — still honour
                if check is not None and target.focused_is_secure(check):   # the secure contract
                    return self._refuse_secure(tgt.app if tgt else "the focused app")
                return True
            if not target.activate_pid(tgt.pid):
                log.warning("could not front %s — pasting into the focused app instead", tgt.app or tgt.pid)
                return True
            time.sleep(0.25)                      # the first responder settles before the ⌘V
            if target.focused_is_secure(tgt.pid):
                return self._refuse_secure(tgt.app)
            self._last_retarget_app = tgt.app or None
            log.info("paste target: %s (under the mouse)", tgt.app)
            return True
        except Exception:
            log.exception("mouse targeting failed — pasting into the focused app")
            return True

    def _refuse_secure(self, where: str) -> bool:
        log.info("secure field focused in %s — leaving the text in the clipboard", where)
        self.emit("status", "Secure field — the text stayed in the clipboard")
        return False

    def _focused_secure(self) -> bool:
        """The secure-field contract for pastes that go to the keyboard focus (no mouse
        targeting): never ⌘V into a native password field."""
        try:
            pid = target.frontmost_pid()
            return pid is not None and target.focused_is_secure(pid)
        except Exception:
            log.exception("secure-field check failed — pasting")
            return False

    # -- retrieval ----------------------------------------------------------
    def paste_block(self, block: Block, retarget: bool = True, source: str = "menu") -> None:
        # Fast lane: pasting is clipboard/⌘V work with zero MLX in it — it must never wait
        # behind a keep-warm touch or a transcription on the model worker queue.
        # retarget=False = paste at the keyboard focus: the caller already fronted the
        # intended app (the window's Paste flow), the pointer is meaningless (menu clicks),
        # or the user pressed Ctrl+Alt+V in the app they are typing in — mouse targeting
        # must not fight that intent.
        with self._paste_spawn_lock:
            if self._repaste_inflight:
                log.debug("re-paste already in flight — dropped (key repeat)")
                return
            self._repaste_inflight = True
        threading.Thread(target=lambda: self._paste_block(block, retarget, source), name="dictdact-paste",
                         daemon=True).start()

    def _paste_block(self, block: Block, retarget: bool = True, source: str = "menu") -> None:
        try:
            with self._paste_lock:
                if retarget:
                    ok = self._mouse_retarget_ok()
                    where = self._last_retarget_app or frontmost_app_name()
                else:
                    where = frontmost_app_name()
                    ok = not self._focused_secure() or self._refuse_secure(where or "the focused app")
                ok = ok and paste_text(block.text, restore_clipboard=self.cfg.restore_clipboard)
                if not ok:
                    set_clipboard_text(block.text)
            self.last_repaste_app = where if ok else None
            log.info("re-paste (%s) → %s%s: %s", source, where or "?", "" if ok else " [clipboard only]",
                     block.preview(60))
            self.emit("retrieved", block)
        except Exception as e:
            log.exception("re-paste failed")
            play("error", self.cfg.sounds)
            self.emit("error", str(e))
        finally:
            with self._paste_spawn_lock:
                self._repaste_inflight = False

    def retrieve_last(self, retarget: bool = True, source: str = "menu") -> None:
        block = self.history.latest()
        if block is None:
            self.emit("status", "No dictation in history yet")
            play("error", self.cfg.sounds)
            return
        self.paste_block(block, retarget, source)

    # -- offline use (CLI / tests) ------------------------------------------------
    def _run_on_worker(self, fn, timeout: float | None = 600.0):
        """Run ``fn`` on the model worker thread. Every MLX call in the process goes through
        that one thread (takes, keep-warm touches, this) — running a model from any other
        thread would race the touches on the shared KV prompt cache."""
        done = threading.Event()
        box: dict = {}

        def job():
            try:
                box["result"] = fn()
            except Exception as e:
                box["error"] = e
            finally:
                done.set()

        self._jobs.put(job)
        if not done.wait(timeout):
            raise TimeoutError("model worker is not responding")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def process_file(self, path: Path, deliver: bool = False) -> tuple[Block | None, object]:
        from .audio import load_wav

        audio = load_wav(path)
        self.wait_ready()
        x, stats = analyze_and_normalize(audio, max_gain_db=self.cfg.max_gain_db)
        if not stats.has_speech:
            return None, stats

        def run_models():
            self._last_model_use = time.time()
            transcript = self.engine.transcribe(x)
            text = collapse_repeats(self.formatter.format(transcript.text)) if transcript.text else ""
            self._last_model_use = time.time()
            return transcript, text

        transcript, text = self._run_on_worker(run_models)
        block = Block.new(text=text, raw=transcript.text, language=transcript.language, engine=transcript.engine,
                          formatter=self.formatter.name, mode=stats.mode, duration_s=round(stats.duration_s, 2))
        if deliver:
            self.history.append(block)
            block.pasted = self._deliver(block.text)
        return block, stats
