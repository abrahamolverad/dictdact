"""Menu-bar UI (rumps) wired to the pipeline and global hotkeys."""
from __future__ import annotations

import logging
import os
import queue
import threading
import subprocess
import time

import rumps

from .audio import list_input_devices
from .config import CONFIG_PATH, HISTORY_PATH, Config
from . import autostart
from .appmenu import MenuController, build_main_menu
from .hotkey import DEFAULT_RETRIEVE_COMBO, Hotkeys, parse_combo, pretty_combo
from .mainwindow import MainWindow
from .server import Backend, Server
from .instance import RUN, STAND_DOWN, ExitGuard, claim_single_instance
from .models import missing_repos
from .setup_window import SetupWindow
from .output import notify, set_clipboard_text
from .overlay import Overlay
from .pipeline import Pipeline
from . import perms

log = logging.getLogger("dictdact.app")

ICON_IDLE, ICON_REC, ICON_LOCK, ICON_BUSY, ICON_LOAD, ICON_ERR = "🎙", "🔴", "🔒", "⏳", "💤", "⚠️"


def _clear(menu: rumps.MenuItem) -> None:
    """rumps raises if you clear a submenu that was never populated."""
    if getattr(menu, "_menu", None) is not None:
        menu.clear()


class DictDactApp(rumps.App):
    def __init__(self, cfg: Config):
        super().__init__("DictDact", title=ICON_LOAD, quit_button=None)
        self.cfg = cfg
        self.events: queue.Queue = queue.Queue()
        self.pipeline = Pipeline(cfg, emit=lambda kind, payload=None: self.events.put((kind, payload)))
        self.hotkeys = self._make_hotkeys()
        self.main_window = None      # created lazily on the main thread
        self._state = "idle"
        self._state_until = 0.0      # when a transient state (done / error) reverts to idle
        self._processing_since = 0.0
        self._status_text = ""
        self.backend = Backend(self)
        self.server = Server(self.backend)
        self._flash_until = 0.0
        self._rec_started = 0.0
        self.overlay = Overlay()
        self._overlay_hide_at = 0.0

        self.status_item = rumps.MenuItem("Loading models…")
        self.mic_menu = rumps.MenuItem("Microphone")
        self.recent_menu = rumps.MenuItem("Recent dictations")
        self.language_menu = rumps.MenuItem("Language")
        self.engine_menu = rumps.MenuItem("Speech engine")
        self.formatter_menu = rumps.MenuItem("Text structuring")
        self.style_menu = rumps.MenuItem("Result style")
        self.paste_item = rumps.MenuItem("Paste into active app", callback=self._toggle_paste)
        self.paste_target_item = rumps.MenuItem("Paste where the mouse points", callback=self._toggle_paste_target)
        self.sounds_item = rumps.MenuItem("Sounds", callback=self._toggle_sounds)
        self.voice_bank_menu = rumps.MenuItem("Voice bank")
        self.paste_item.state = int(cfg.paste)
        self.paste_target_item.state = int(cfg.paste_target == "mouse")
        self.sounds_item.state = int(cfg.sounds)
        self.autostart_item = rumps.MenuItem("Start at login & restart if it crashes", callback=self._toggle_autostart)
        self.autostart_item.state = int(autostart.enabled()) if autostart.available() else 0
        repaste = self._repaste_label()
        help_item = rumps.MenuItem(f"Hold {cfg.hotkey.upper()} to dictate · tap-tap to lock · {repaste} pastes the last take where the cursor is · Ctrl+Alt+H window")
        self.menu = [
            self.status_item,
            help_item,
            None,
            self.mic_menu,
            self.recent_menu,
            None,
            self.language_menu,
            rumps.MenuItem("Vocabulary…  (names & tools to spell right)", callback=lambda _: self._show_window("&tab=vocab")),
            self.engine_menu,
            self.formatter_menu,
            self.style_menu,
            self.paste_item,
            self.paste_target_item,
            self.sounds_item,
            self.voice_bank_menu,
            *([self.autostart_item] if autostart.available() else []),
            None,
            rumps.MenuItem("Open DictDact…  (Ctrl+Alt+H)", callback=lambda _: self._show_window()),
            rumps.MenuItem("Guía rápida / Quick guide", callback=lambda _: self._show_window("&tab=guide")),
            rumps.MenuItem(f"Paste last dictation  ({repaste})",
                           callback=lambda _: self.pipeline.retrieve_last(retarget=False, source="menu")),
            rumps.MenuItem("Copy last dictation", callback=self._copy_last),
            rumps.MenuItem("Open history", callback=lambda _: subprocess.Popen(["open", "-t", str(HISTORY_PATH)])),
            rumps.MenuItem("Open config", callback=lambda _: subprocess.Popen(["open", "-t", str(CONFIG_PATH)])),
            rumps.MenuItem("Check permissions…", callback=self._check_perms),
            None,
            rumps.MenuItem("Quit DictDact", callback=self._quit),
        ]
        self._build_mic_menu()
        self._build_voice_bank_menu()
        self._build_language_menu()
        self._build_choice_menu(self.engine_menu, ["whisper", "parakeet"], cfg.engine, self._pick_engine)
        self._build_choice_menu(self.formatter_menu, ["mlx", "ollama", "rules", "off"], cfg.formatter, self._pick_formatter)
        from .format import STYLES
        self._build_choice_menu(self.style_menu, list(STYLES), cfg.formatter_style, self._pick_style)
        self._build_recent_menu()
        self.timer = rumps.Timer(self._tick, 0.2)
        self.fast_timer = rumps.Timer(self._fast_tick, 1 / 30)
        self.perm_timer = rumps.Timer(self._perm_tick, 3)
        self._had_input_monitoring = None
        self.setup = None
        self.setup_timer = rumps.Timer(self._setup_tick, 0.5)
        self._setup_close_at = 0.0
        self.exit_guard = ExitGuard()
        self.menu_controller = None
        self._started_at = time.time()
        self.twin_timer = rumps.Timer(self._twin_check, 2)   # closes the simultaneous-start race
        self._twin_checks = 0

    # -- menus ------------------------------------------------------------------
    def _build_mic_menu(self) -> None:
        _clear(self.mic_menu)
        current = self.pipeline.recorder.device.name if self.pipeline.recorder and self.pipeline.recorder.device else None
        item = rumps.MenuItem("System default", callback=lambda _: self._pick_mic(None))
        item.state = int(self.cfg.microphone is None)
        self.mic_menu.add(item)
        for d in list_input_devices():
            label = d.name + ("  (default)" if d.is_default else "")
            item = rumps.MenuItem(label, callback=lambda _, name=d.name: self._pick_mic(name))
            item.state = int(self.cfg.microphone is not None and d.name == current)
            self.mic_menu.add(item)
        self.mic_menu.add(None)
        self.mic_menu.add(rumps.MenuItem("Refresh devices", callback=lambda _: self._build_mic_menu()))

    def _build_choice_menu(self, menu, options, current, cb) -> None:
        _clear(menu)
        for opt in options:
            item = rumps.MenuItem(opt, callback=lambda _, o=opt: cb(o))
            item.state = int(opt == current)
            menu.add(item)

    LANGUAGES = (("Auto (Spanish + English)", None), ("Español — locked", "es"), ("English — locked", "en"))

    def _build_language_menu(self) -> None:
        _clear(self.language_menu)
        current = self.cfg.language if self.cfg.language in ("es", "en") else None
        for label, code in self.LANGUAGES:
            item = rumps.MenuItem(label, callback=lambda _, c=code: self._pick_language(c))
            item.state = int(code == current)
            self.language_menu.add(item)
        self.language_menu.title = "Language: " + {"es": "Español", "en": "English"}.get(current, "Auto")

    def _pick_language(self, code):
        self.pipeline.set_language(code)
        self._build_language_menu()

    def _build_recent_menu(self) -> None:
        _clear(self.recent_menu)
        blocks = self.pipeline.history.last(12)
        if not blocks:
            empty = rumps.MenuItem("(nothing yet)")
            empty.set_callback(None)
            self.recent_menu.add(empty)
            return
        for b in blocks:
            label = f"{b.ts[11:16]}  {b.preview(70)}"
            # retarget=False: the pointer is inside this menu at the top of the screen —
            # mouse targeting would front whatever app sits behind it.
            self.recent_menu.add(rumps.MenuItem(label, callback=lambda _, blk=b: self.pipeline.paste_block(blk, retarget=False)))

    # -- callbacks --------------------------------------------------------------
    def _pick_mic(self, name):
        self.pipeline.set_microphone(name)
        self._build_mic_menu()
        self._status(f"Mic: {name or 'system default'}")

    def _pick_engine(self, engine):
        self.pipeline.set_engine(engine)
        self._build_choice_menu(self.engine_menu, ["whisper", "parakeet"], engine, self._pick_engine)

    def _pick_formatter(self, fmt):
        self.pipeline.set_formatter(fmt)
        self._build_choice_menu(self.formatter_menu, ["mlx", "ollama", "rules", "off"], fmt, self._pick_formatter)

    def _pick_style(self, style):
        from .format import STYLES
        self.pipeline.set_formatter_style(style)
        self._build_choice_menu(self.style_menu, list(STYLES), style, self._pick_style)
        self._status(f"Result style: {style}" + ("" if style == "clean" else " — dictations will be transformed"))

    def _toggle_paste(self, item):
        self.cfg.paste = not self.cfg.paste
        self.cfg.save()
        item.state = int(self.cfg.paste)

    def _toggle_paste_target(self, item):
        self.cfg.paste_target = "focus" if self.cfg.paste_target == "mouse" else "mouse"
        self.cfg.save()
        item.state = int(self.cfg.paste_target == "mouse")

    def _toggle_sounds(self, item):
        self.cfg.sounds = not self.cfg.sounds
        self.cfg.save()
        item.state = int(self.cfg.sounds)

    def _toggle_voice_bank(self, item):
        self.cfg.voice_bank = not self.cfg.voice_bank
        self.cfg.save()
        if not self.cfg.voice_bank:
            self.pipeline.voicebank.switched_off()   # returns once nothing more can be saved
        self._build_voice_bank_menu()
        self._status("Voice bank on — your takes are kept for voice cloning" if self.cfg.voice_bank
                      else "Voice bank off")

    def _build_voice_bank_menu(self) -> None:
        """Voice bank submenu: is it recording, how far each language is toward ElevenLabs, what
        happened to the latest take, how the room sounds, and one-click exports."""
        from . import voicebank as vb

        try:
            view = vb.menu_view(vb.summary(), on=bool(getattr(self.cfg, "voice_bank", False)),
                                last=self.pipeline.voicebank.last,
                                target_min=float(getattr(self.cfg, "voice_bank_target_min", 180.0)))
        except Exception:
            log.exception("voice bank menu")
            view = {"title": "Voice bank", "lines": ["(could not read the bank)"], "exportable": {}}
        self.voice_bank_menu.title = view["title"]
        _clear(self.voice_bank_menu)
        toggle = rumps.MenuItem("Record my voice for cloning (ElevenLabs)", callback=self._toggle_voice_bank)
        toggle.state = int(bool(getattr(self.cfg, "voice_bank", False)))
        self.voice_bank_menu.add(toggle)
        self.voice_bank_menu.add(None)
        for line in view["lines"]:
            self.voice_bank_menu.add(rumps.MenuItem(line, callback=lambda _: None))   # readable, not greyed
        self.voice_bank_menu.add(None)
        for lang in vb.LANGS:
            name = vb.LANG_NAMES[lang]
            if view["exportable"].get(lang):
                item = rumps.MenuItem(f"Export {name} for ElevenLabs…", callback=lambda _, l=lang: self._export_voice(l))
            else:
                item = rumps.MenuItem(f"Export {name} for ElevenLabs  (no usable minutes yet)")   # greyed out
            self.voice_bank_menu.add(item)
        self.voice_bank_menu.add(rumps.MenuItem("Open voice bank folder", callback=lambda _: self._open_voice_bank()))

    def _open_voice_bank(self) -> None:
        from . import voicebank as vb

        vb.bank_dir().mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open", str(vb.bank_dir())])

    def _export_voice(self, lang: str) -> None:
        """Exports can take a while (hours of audio → MP3): off the main thread, then Finder."""
        from . import voicebank as vb

        name = vb.LANG_NAMES[lang]
        self._status(f"Exporting {name} for ElevenLabs…")

        def run():
            try:
                r = vb.export(lang)
            except Exception as e:
                log.exception("voice bank export")
                notify("Voice bank", f"{name} export failed: {e}")
                return
            subprocess.Popen(["open", r["dir"]])
            short = "" if r["minutes"] >= vb.GOAL_MIN else f" — ElevenLabs needs {vb.GOAL_MIN} min for a Professional Voice Clone"
            notify("Voice bank", f"{name}: {r['minutes']} min in {len(r['files'])} file(s), folder opened{short}")
            log.info("voice bank: exported %s → %s", lang, r["dir"])

        threading.Thread(target=run, name="dictdact-voice-export", daemon=True).start()

    def _apply_dock_icon(self) -> None:
        """A Dock icon (and ⌘-Tab entry) makes DictDact a normal app you can always reopen.
        Done at runtime, so the bundle's signature — and its macOS permissions — stay untouched."""
        try:
            from AppKit import (NSApp, NSApplicationActivationPolicyAccessory,
                                NSApplicationActivationPolicyRegular)

            want = NSApplicationActivationPolicyRegular if self.cfg.dock_icon else NSApplicationActivationPolicyAccessory
            if NSApp is None:
                return
            if NSApp.activationPolicy() != want:
                NSApp.setActivationPolicy_(want)
                log.info("dock icon %s", "on" if self.cfg.dock_icon else "off")
            if self.cfg.dock_icon:
                if self.menu_controller is None:
                    self.menu_controller = MenuController.alloc().initWithApp_(self)
                if NSApp.mainMenu() is None:
                    NSApp.setMainMenu_(build_main_menu(self.menu_controller))
        except Exception:
            log.exception("could not apply the dock icon setting")

    def _observe_activation(self) -> None:
        """Clicking the Dock icon (or ⌘-Tab) with no window open should open the window."""
        try:
            from Foundation import NSNotificationCenter, NSObject
            import objc

            app = self

            class _DictDactActivationWatcher(NSObject):
                def becameActive_(self, note):
                    app._on_activated()

            self._activation_watcher = _DictDactActivationWatcher.alloc().init()
            NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
                self._activation_watcher, "becameActive:", "NSApplicationDidBecomeActiveNotification", None)
        except Exception:
            log.exception("could not observe activation")

    def _on_activated(self) -> None:
        if time.time() - self._started_at < 5 and not self.cfg.show_window_at_launch:
            return   # the launch activation itself should not pop a window in your face
        if self.main_window is None or not self.main_window.is_visible():
            self._show_window()

    def _show_window(self, path: str = ""):
        if self.main_window is None:
            self.main_window = MainWindow.alloc().initWithURL_(self.server.url)
        self.main_window.show(path)
        log.info("window shown (%s)", path or "default tab")

    def _make_hotkeys(self) -> Hotkeys:
        # Ctrl+Alt+V = the recovery key: paste the last take where the KEYBOARD cursor is.
        # The user is pressing keys in the app they mean; mouse targeting (which may be what
        # sent the dictation to the wrong window in the first place) must not apply here.
        label = self._repaste_label()
        return Hotkeys(
            self.cfg.hotkey, self.cfg.hold_threshold_ms, self.cfg.double_tap_lock,
            on_start=self.pipeline.start_recording, on_stop=self.pipeline.stop_recording,
            on_cancel=self.pipeline.cancel_recording, on_lock=lambda: self.events.put(("locked", None)),
            on_retrieve=lambda: self.pipeline.retrieve_last(retarget=False, source=label),
            retrieve_combo=self._repaste_combo(),
            extra_combos={"ctrl+alt+h": lambda: self.events.put(("show_history", None))})

    def _repaste_combo(self) -> str:
        """The configured re-paste shortcut, or the default when it does not parse. A typo in
        config.json must not take the app down with it — dictation still has to work."""
        spec = getattr(self.cfg, "retrieve_combo", None) or DEFAULT_RETRIEVE_COMBO
        try:
            parse_combo(spec)
        except ValueError as e:
            log.warning("retrieve_combo: %s — falling back to %s", e, DEFAULT_RETRIEVE_COMBO)
            return DEFAULT_RETRIEVE_COMBO
        return spec

    def _repaste_label(self) -> str:
        return pretty_combo(self._repaste_combo())

    def _reload_hotkeys(self):
        if self.pipeline.recorder and self.pipeline.recorder.recording:
            self.hotkeys.force_stop_recording()   # finish the take under the old key first
        try:
            self.hotkeys.stop()
        except Exception:
            pass
        self.hotkeys = self._make_hotkeys()
        self.hotkeys.start()
        self._status(f"Hotkey is now {self.cfg.hotkey.upper()}")

    def _toggle_autostart(self, item):
        on = not bool(item.state)
        if autostart.set_enabled(on):
            item.state = int(on)
            if on and autostart.requires_approval():
                self._status("Approve DictDact under System Settings → General → Login Items", flash_s=10)
                subprocess.Popen(["open", "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"])

    def _quit(self, _=None):
        log.info("user quit from the menu")
        self.exit_guard.user_quit = True
        rumps.quit_application()

    def _install_quit_signal(self):
        """SIGUSR1 = quit cleanly, exactly like the menu (used by `dictdact quit`)."""
        import signal

        def handler(signum, frame):
            log.info("SIGUSR1 received — clean quit")
            self._quit()

        signal.signal(signal.SIGUSR1, handler)
        signal.signal(signal.SIGUSR2, lambda s, f: self.events.put(("show_history", None)))  # `dictdact open`

    def _copy_last(self, _):
        b = self.pipeline.history.latest()
        if b:
            set_clipboard_text(b.text)
            self._status("Copied last dictation")

    def _check_perms(self, _):
        lines = []
        for name, ok, where in perms.report():
            lines.append(f"{'✅' if ok else '❌'} {name}" + ("" if ok else f"\n     → {where}"))
        rumps.alert("DictDact permissions", "\n".join(lines) +
                    "\n\nGrant them to the app you launched DictDact from (Terminal, iTerm, Claude…) "
                    "or to DictDact.app, then restart DictDact.")

    # -- UI updates (main thread) -------------------------------------------------
    def _status(self, text: str, flash_s: float = 4.0) -> None:
        self.status_item.title = text
        self._status_text = text
        self._flash_until = time.time() + flash_s

    def _set_state(self, state: str, for_s: float = 0.0) -> None:
        self._state = state
        self._state_until = time.time() + for_s if for_s else 0.0

    def _mic_short(self) -> str:
        rec = self.pipeline.recorder
        name = rec.device.name if rec and rec.device else "default mic"
        return name if len(name) <= 16 else name.split()[0]

    def _overlay_flash(self, state: str, label: str, seconds: float) -> None:
        self.overlay.show(state, label)
        self._overlay_hide_at = time.time() + seconds

    def _perm_tick(self, _):
        """While a permission is missing, keep checking; when Input Monitoring shows up,
        restart the keyboard listener so the user doesn't have to relaunch."""
        try:
            im, ax = perms.input_monitoring(), perms.accessibility()
            if self._had_input_monitoring is False and im:
                log.info("Input Monitoring granted — restarting hotkey listener")
                self.hotkeys.restart()
                self._status("Keyboard access granted — hold Ctrl to dictate")
            self._had_input_monitoring = im
            if im and ax:
                self.perm_timer.stop()
                self._status("Ready")
            else:
                missing = [n for n, ok in (("Input Monitoring", im), ("Accessibility", ax)) if not ok]
                self._status("Grant " + " + ".join(missing) + " to DictDact in System Settings", flash_s=0)
        except Exception:
            log.exception("perm tick failed")

    def _fast_tick(self, _):
        try:
            self._fast_tick_inner()
        except Exception:
            if time.time() - getattr(self, "_last_tick_err", 0) > 10:
                self._last_tick_err = time.time()
                log.exception("overlay tick failed")

    def _fast_tick_inner(self):
        """30 fps: feed the waveform, reveal the pill once a hold is confirmed, expire flashes."""
        rec = self.pipeline.recorder
        now = time.time()
        if rec and rec.recording:
            self._overlay_hide_at = 0.0
            if not self.overlay.visible and now - self._rec_started >= self.cfg.hold_threshold_ms / 1000.0:
                state = "locked" if self.hotkeys.locked else "recording"
                label = "Locked · tap Ctrl to finish" if state == "locked" else f"Listening · {self._mic_short()}"
                self.overlay.show(state, label)
            if self.overlay.visible:
                self.overlay.push_level(rec.latest_rms())
                if rec.seconds > 1.5 and rec.max_rms() == 0.0 and self.overlay.view.state != "nosignal":
                    self.overlay.set_state("nosignal", f"No signal from {self._mic_short()} — check mic")
        elif self._overlay_hide_at and now > self._overlay_hide_at:
            self._overlay_hide_at = 0.0
            self.overlay.hide()
        self.overlay.tick()

    def _tick(self, _):
        try:
            self._tick_inner()
        except Exception:
            log.exception("ui tick failed")

    def _tick_inner(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "ready":
                self.title = ICON_IDLE
                self._status(f"Ready · {payload}", flash_s=0)
            elif kind == "show_history":
                self._show_window()
            elif kind == "reload_hotkey":
                self._reload_hotkeys()
            elif kind == "dock_changed":
                self._apply_dock_icon()
            elif kind == "config_changed":
                self.paste_item.state = int(self.cfg.paste)
                self.paste_target_item.state = int(self.cfg.paste_target == "mouse")
                self.sounds_item.state = int(self.cfg.sounds)
                self._build_voice_bank_menu()
                self._build_mic_menu()
                self._build_language_menu()
                self._build_choice_menu(self.engine_menu, ["whisper", "parakeet"], self.cfg.engine, self._pick_engine)
                self._build_choice_menu(self.formatter_menu, ["mlx", "ollama", "rules", "off"], self.cfg.formatter, self._pick_formatter)
                from .format import STYLES
                self._build_choice_menu(self.style_menu, list(STYLES), self.cfg.formatter_style, self._pick_style)
            elif kind == "paste_from_window":
                # returned = the window really fronted a previous app; the paste must then
                # honour that explicit intent — mouse targeting would fight it (the pointer
                # still hovers where the DictDact window was, over whatever sits behind it).
                # A window opened while DictDact was already frontmost has no prev_app; the
                # mouse correction stays on for that case.
                returned = self.main_window.return_to_previous_app() if self.main_window is not None else False
                threading.Timer(0.45, lambda: self.pipeline.paste_block(payload, retarget=not returned)).start()
            elif kind == "status":
                self._status(str(payload))
                if self.overlay.visible and self.overlay.view.state == "processing":
                    self.overlay.set_state("processing", str(payload))
            elif kind == "recording":
                self._rec_started = time.time()
                self._state = "recording"
                self.title = ICON_REC
                self._status("Listening…", flash_s=0)
            elif kind == "locked":
                self._state = "locked"
                self.title = ICON_LOCK
                self._status("Locked — tap Ctrl to finish", flash_s=0)
                if self.overlay.visible:
                    self.overlay.set_state("locked", "Locked · tap Ctrl to finish")
            elif kind == "processing":
                self._state = "processing"
                self._processing_since = time.time()
                self.title = ICON_BUSY
                self._status("Transcribing…", flash_s=0)
                self._overlay_hide_at = 0.0
                self.overlay.show("processing", "Transcribing…")
            elif kind == "idle":
                self._state = "idle"
                self.title = ICON_IDLE
                self._status(str(payload) if payload else "Ready")
                if payload:
                    self._overlay_flash("error", str(payload), 3.5 if "low" in str(payload) else 1.8)
                else:
                    self.overlay.hide()
            elif kind == "done":
                self._set_state("done", 2.0)
                self.title = ICON_IDLE
                tag = "" if payload.pasted else " (copied — paste blocked?)"
                self._status(f"{payload.mode}: {payload.preview(60)}{tag}")
                label = "Pasted ✓" if payload.pasted else "Copied ✓ — paste blocked"
                if payload.mode == "whisper":
                    label += "  · whisper-level input"
                self._overlay_flash("done", label, 1.6 if payload.mode == "whisper" else 1.0)
                self._build_recent_menu()
            elif kind == "voicebank":
                self._build_voice_bank_menu()        # the bank finished with a take (saved or skipped)
            elif kind == "retrieved":
                self._set_state("done", 2.0)
                self.title = ICON_IDLE
                where = self.pipeline.last_repaste_app
                if where:
                    self._status(f"Re-pasted into {where}: {payload.preview(50)}")
                    self._overlay_flash("done", f"Re-pasted ✓ → {where}", 1.2)
                else:
                    self._status(f"Copied (paste blocked): {payload.preview(50)}")
                    self._overlay_flash("done", "Copied ✓ — paste blocked", 1.6)
            elif kind == "error":
                self._set_state("error", 5.0)
                self.title = ICON_ERR
                self._status(f"Error: {payload}", flash_s=10)
                self._overlay_flash("error", str(payload)[:70], 4.0)
                notify("DictDact", str(payload))
        if self._state_until and time.time() > self._state_until and self._state in ("done", "error"):
            self._set_state("idle")
        if self._state == "processing" and time.time() - self._processing_since > 120:
            # Nothing should take two minutes. Exit 1 → launchd restarts us in ~5 s.
            log.error("take stuck in processing for 120 s — restarting")
            os._exit(1)
        # live timer while recording + time-cap enforcement
        rec = self.pipeline.recorder
        if rec and rec.recording:
            self.status_item.title = f"Listening… {rec.seconds:0.0f}s   {rec.level_dbfs():.0f} dBFS"
            if rec.hit_cap:
                self.hotkeys.force_stop_recording()
        elif self._flash_until and time.time() > self._flash_until:
            self._flash_until = 0.0
            self.status_item.title = "Ready" if self.pipeline.ready else self.status_item.title

    # -- lifecycle ----------------------------------------------------------------
    def _disable_app_nap(self) -> None:
        """A windowless agent gets App-Napped (CPU/GPU throttled) while idle — exactly when a
        dictation arrives. Hold a user-initiated, latency-critical activity for our lifetime."""
        try:
            from Foundation import NSProcessInfo, NSActivityUserInitiated, NSActivityLatencyCritical

            self._activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                NSActivityUserInitiated | NSActivityLatencyCritical, "DictDact dictation")
            log.info("app nap disabled")
        except Exception as e:
            log.warning("could not disable app nap: %s", e)

    def _twin_check(self, _):
        """Second ownership check after we are visible to LaunchServices: two copies that started
        at the same instant could each have seen an empty registry the first time."""
        try:
            self._twin_checks += 1
            if self._twin_checks >= 3:
                self.twin_timer.stop()
            verdict = claim_single_instance()
            if verdict != RUN:
                log.warning("twin detected after start — this copy exits (%s)", verdict)
                os._exit(0 if verdict == STAND_DOWN else 1)
        except Exception:
            log.exception("twin check failed")

    def _setup_tick(self, _):
        try:
            if self.setup is None:
                return
            self.setup.refresh()
            if self._setup_close_at and time.time() > self._setup_close_at:
                self.setup.close()
                self.setup_timer.stop()
        except Exception:
            log.exception("setup tick failed")

    def _on_setup_ready(self):
        log.info("first-run setup complete")
        if autostart.available() and not autostart.enabled():
            if autostart.set_enabled(True):
                self.autostart_item.state = 1
                log.info("autostart enabled (login item)")
        self.pipeline.load_models()
        self._setup_close_at = time.time() + 2.5
        if perms.input_monitoring() and self._had_input_monitoring is False:
            self.hotkeys.restart()

    def start(self) -> None:
        self._started_at = time.time()
        self.exit_guard.install()
        self._install_quit_signal()
        self.server.start()
        self._disable_app_nap()
        missing = missing_repos(self.cfg)
        mic = perms.microphone()
        log.info("permissions: input_monitoring=%s accessibility=%s microphone=%s | models missing: %s",
                 perms.input_monitoring(), perms.accessibility(), mic, missing or "none")
        if perms.input_monitoring() is False:
            # macOS 26: Accessibility no longer covers key observation. Ask the system
            # directly — the prompt registers a correctly-named row in Input Monitoring
            # (hunting the pane by hand can surface the bundle's inner python binary).
            perms.input_monitoring(prompt=True)
        if missing or perms.accessibility() is False or mic == "denied":
            # First run (or a lost grant): guide the user instead of silently doing nothing.
            self.setup = SetupWindow.alloc().initWithRepos_onReady_(missing, self._on_setup_ready)
            self.setup.show()
            self.setup_timer.start()
            self._status("Finish the steps in the Welcome window", flash_s=0)
        else:
            perms.request_microphone(lambda ok: log.info("microphone access %s", "granted" if ok else "DENIED"))
            self.pipeline.load_models()
        try:
            self.hotkeys.start()
        except Exception as e:
            log.error("hotkeys unavailable: %s", e)
            self.events.put(("error", "Hotkey listener failed — grant Input Monitoring and restart"))
        self._apply_dock_icon()
        self._observe_activation()
        self.timer.start()
        self.fast_timer.start()
        self.twin_timer.start()
        self._had_input_monitoring = perms.input_monitoring()
        if not (self._had_input_monitoring and perms.accessibility()):
            self.perm_timer.start()
        if self.cfg.show_window_at_launch:
            self.events.put(("show_history", None))
        if perms.input_monitoring() is False or perms.accessibility() is False:
            notify("DictDact", "Grant Input Monitoring + Accessibility to the app that launched DictDact, then restart.")
        self.run()


def run_headless(cfg: Config) -> None:
    """Same pipeline, no menu bar — prints events to the terminal."""
    def emit(kind, payload=None):
        if kind == "done":
            print(f"\n[{payload.mode}] {payload.text}\n", flush=True)
        elif kind in ("status", "idle", "error", "ready"):
            print(f"[{kind}] {payload or ''}", flush=True)
        elif kind in ("recording", "processing", "locked", "retrieved"):
            print(f"[{kind}]", flush=True)

    pipeline = Pipeline(cfg, emit=emit)
    pipeline.load_models()
    combo = getattr(cfg, "retrieve_combo", None) or DEFAULT_RETRIEVE_COMBO
    try:
        label = pretty_combo(combo)
    except ValueError as e:
        log.warning("retrieve_combo: %s — falling back to %s", e, DEFAULT_RETRIEVE_COMBO)
        combo, label = DEFAULT_RETRIEVE_COMBO, pretty_combo(DEFAULT_RETRIEVE_COMBO)
    hk = Hotkeys(cfg.hotkey, cfg.hold_threshold_ms, cfg.double_tap_lock,
                 on_start=pipeline.start_recording, on_stop=pipeline.stop_recording,
                 on_cancel=pipeline.cancel_recording, on_lock=lambda: emit("locked"),
                 on_retrieve=lambda: pipeline.retrieve_last(retarget=False, source=label),
                 retrieve_combo=combo)
    hk.start()
    print(f"Hold {cfg.hotkey} to dictate. {label} re-pastes the last block. Ctrl+C here to quit.", flush=True)
    try:
        while True:
            time.sleep(0.2)
            if pipeline.recorder and pipeline.recorder.hit_cap:
                hk.force_stop_recording()
    except KeyboardInterrupt:
        hk.stop()
