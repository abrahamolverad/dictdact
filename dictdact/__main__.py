from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .config import CONFIG_PATH, HISTORY_PATH, Config
from .log import setup_logging


def cmd_run(args, cfg: Config) -> None:
    cfg.ensure_dirs()
    if args.headless:
        from .app import run_headless

        run_headless(cfg)
    else:
        from AppKit import NSApplication

        from .app import DictDactApp
        from .instance import claim_single_instance

        from .instance import RUN, STAND_DOWN

        NSApplication.sharedApplication()  # register with LaunchServices BEFORE looking for twins
        verdict = claim_single_instance()
        if verdict != RUN:
            from .output import notify

            notify("DictDact", "DictDact is already running.")
            sys.exit(0 if verdict == STAND_DOWN else 1)  # stand-down = stay down; yield = launchd may retry
        DictDactApp(cfg).start()


def cmd_devices(args, cfg: Config) -> None:
    from .audio import list_input_devices

    for d in list_input_devices():
        mark = "*" if d.is_default else " "
        sel = "  ← selected" if cfg.microphone and cfg.microphone.lower() in d.name.lower() else ""
        print(f"{mark} [{d.index}] {d.name}  ({d.channels} ch, {d.default_samplerate:.0f} Hz){sel}")
    print("\n* = system default.  Select one with:  dictdact config --mic 'Name'")


def cmd_transcribe(args, cfg: Config) -> None:
    from .pipeline import Pipeline

    pipeline = Pipeline(cfg)
    pipeline.load_models()
    pipeline.wait_ready()
    for f in args.files:
        block, stats = pipeline.process_file(Path(f), deliver=args.paste)
        print(f"\n=== {f}\naudio: {stats.summary()}")
        if block is None:
            print("(no speech detected)")
            continue
        print(f"raw  [{block.language}] ({block.engine}): {block.raw}")
        print(f"text [{block.formatter}]:\n{block.text}")


def cmd_history(args, cfg: Config) -> None:
    from .history import History

    blocks = History().last(args.n)
    if not blocks:
        print(f"(no dictations yet — {HISTORY_PATH})")
    for b in blocks:
        flag = "" if b.pasted else "  [not pasted]"
        print(f"--- {b.ts}  {b.mode} {b.duration_s}s  {b.app or ''}{flag}\n{b.text}\n")


def cmd_doctor(args, cfg: Config) -> None:
    from . import perms
    from .audio import list_input_devices, resolve_input_device

    print("Permissions (for THIS process / the app that launched it):")
    for name, ok, where in perms.report():
        print(f"  {'✅' if ok else '❌'} {name}" + ("" if ok else f"   → {where}"))
    dev = resolve_input_device(cfg.microphone)
    print(f"\nMicrophone: {dev.name if dev else 'none found'}   (config: {cfg.microphone or 'system default'})")
    print(f"Engine: {cfg.engine} ({cfg.whisper_model if cfg.engine == 'whisper' else cfg.parakeet_model})")
    print(f"Formatter: {cfg.formatter}")
    if cfg.formatter == "ollama":
        import urllib.request

        try:
            urllib.request.urlopen(f"{cfg.ollama_url}/api/tags", timeout=2).read()
            print("  ollama: reachable")
        except Exception as e:
            print(f"  ollama: NOT reachable ({e})")
    from huggingface_hub import scan_cache_dir

    cached = {r.repo_id for r in scan_cache_dir().repos}
    for repo in (cfg.whisper_model, cfg.parakeet_model, cfg.mlx_formatter_model):
        print(f"  model {'cached' if repo in cached else 'MISSING'}: {repo}")
    print(f"\nConfig:  {CONFIG_PATH}\nHistory: {HISTORY_PATH}")


def cmd_config(args, cfg: Config) -> None:
    changed = False
    if args.mic is not None:
        cfg.microphone = None if args.mic.lower() in ("default", "none", "") else args.mic
        changed = True
    for key in ("engine", "formatter", "hotkey", "language"):
        val = getattr(args, key)
        if val is not None:
            if key == "language" and val not in ("auto", "es", "en"):
                raise SystemExit("language must be 'auto', 'es' or 'en'")
            setattr(cfg, key, None if val == "auto" else val)
            changed = True
    if args.set:
        for pair in args.set:
            k, _, v = pair.partition("=")
            if not hasattr(cfg, k):
                raise SystemExit(f"unknown config key {k!r}")
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                v = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            if k == "formatter_style":
                from .format import STYLES

                if v not in STYLES:
                    raise SystemExit(f"formatter_style must be one of: {', '.join(STYLES)}")
            setattr(cfg, k, v)
            changed = True
    if changed:
        cfg.save()
        print(f"saved {CONFIG_PATH}")
        if args.language is not None and _notify_running_app({"language": cfg.language}, "/api/config"):
            print("the running DictDact switched language live")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))


def _notify_running_app(body: dict, path: str = "/api/vocab") -> bool:
    """POST to the running app's localhost API (token from ui-url). False when it is not running."""
    import json as _json
    import urllib.request

    from .config import DATA_DIR

    p = DATA_DIR / "ui-url"
    if not p.exists():
        return False
    url = p.read_text().strip()
    base, _, tok = url.partition("?t=")
    req = urllib.request.Request(base.rstrip("/") + path, data=_json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-DictDact-Token": tok})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            r.read()
        return True
    except Exception:
        return False


def cmd_vocab(args, cfg: Config) -> None:
    """The dictionary: names, products and tools written exactly the way you want."""
    from .vocab import VOCAB_PATH, Vocabulary, vocab_lock

    with vocab_lock():                      # load → modify → save as one step, even beside the app
        v = Vocabulary.load()
        if args.vcmd == "add":
            spoken = [s.strip() for s in (args.spoken or "").split(",") if s.strip()]
            t = v.add(args.term, spoken, args.note or "", fuzzy=(True if args.fuzzy else (False if args.no_fuzzy else None)))
            v.save()
            print(f"added {t.term}" + (f"  (spoken as: {', '.join(t.spoken)})" if t.spoken else ""))
        elif args.vcmd == "remove":
            if not v.remove(args.term):
                raise SystemExit(f"not in the dictionary: {args.term}")
            v.save()
            print(f"removed {args.term}")
    if args.vcmd in ("test", "suggest"):
        from .history import History
        from .learn import protected_words, suggest

        blocks = [b for b in History().all() if b.raw]
        raws = [b.raw for b in blocks]
        v.protected = protected_words(raws)            # exactly what the running app does
    if args.vcmd == "test":
        print(v.apply(" ".join(args.text)))
        return
    if args.vcmd == "suggest":
        items = suggest(raws, v, limit=args.n, texts=[b.text for b in blocks if b.text])
        if not items:
            print(f"nothing to suggest from {len(raws)} takes")
            return
        print(f"names your takes spell more than one way ({len(raws)} takes):")
        for s in items:
            ev = ", ".join(f"{w} ×{n}" for w, n in s.spellings.items())
            print(f"  {s.term:<14} ← {', '.join(s.spoken):<30} [{ev}]")
        print('\nadd one with:  dictdact vocab add "Hyrox" --spoken "Hirox, Hierox"')
        return
    if args.vcmd not in ("add", "remove"):
        print(f"{len(v)} terms in {VOCAB_PATH}\n")
        for t in v.terms:
            extra = ("  ← " + ", ".join(t.spoken)) if t.spoken else ""
            note = f"   [{t.note}]" if t.note else ""
            print(f"  {t.term}{extra}{note}")
        return
    if _notify_running_app({"reload": True}):
        print("the running DictDact picked it up")
    else:
        print("DictDact is not running — it will load the dictionary at the next launch")


def cmd_voicebank(args, cfg: Config) -> None:
    """Your own voice, collected from everyday dictation, for a voice clone."""
    from . import voicebank as vb

    sub = args.bcmd or "status"
    if sub in ("on", "off"):
        on = sub == "on"
        if _notify_running_app({"voice_bank": on}, "/api/config"):
            print(f"voice bank {sub} (live in the running DictDact)")
        else:
            cfg.voice_bank = on
            cfg.save()
            print(f"voice bank {sub} — DictDact is not running; it applies at the next launch")
        return
    if sub == "export":
        try:
            r = vb.export(args.lang, minutes=args.minutes, out_dir=args.out, fmt=args.format,
                          chunk_min=args.chunk_min, min_snr=args.min_snr)
        except (ValueError, RuntimeError) as e:
            raise SystemExit(f"export: {e}")
        print(f"{r['minutes']} min of {r['language']} from {r['takes']} takes ({r['device']}), "
              f"median SNR {r['median_snr_db']:.0f} dB → {len(r['files'])} file(s) in\n  {r['dir']}")
        if r["excluded"]:
            print("left out: " + ", ".join(f"{k} ×{v}" for k, v in r["excluded"].items()))
        if r["minutes"] < 30:
            print("note: ElevenLabs' Professional Voice Clone needs at least 30 min (2–3 h is best)")
        if args.open:
            import subprocess

            subprocess.Popen(["open", r["dir"]])
        return
    if sub == "reference":
        try:
            print(vb.reference(args.lang, seconds=args.seconds, out=args.out, min_snr=args.min_snr))
        except RuntimeError as e:
            raise SystemExit(f"reference: {e}")
        return
    if sub == "say":
        try:
            print(vb.say(" ".join(args.text), args.lang, out=args.out, min_snr=args.min_snr))
        except RuntimeError as e:
            raise SystemExit(f"say: {e}")
        return
    st = vb.summary(min_snr=args.min_snr)
    state = "ON" if cfg.voice_bank else "off"
    print(f"voice bank: {state} · {st['takes']} takes · {st['stored_s'] / 3600:.1f} h stored in {vb.bank_dir()}")
    if not st["langs"]:
        print("nothing banked yet" + ("" if cfg.voice_bank else " — turn it on with: dictdact voicebank on"))
        return
    for lang, s in st["langs"].items():
        goals = [g for g in vb.MILESTONES_MIN if s["usable_min"] < g]
        if s["per_day_min"] <= 0:
            eta = "no usable takes yet — see the verdict below"
        else:
            eta = " · ".join(f"{g} min in ~{(g - s['usable_min']) / s['per_day_min']:.1f} d" for g in goals)
        if s["usable_min"] >= 30:
            eta = "✓ enough for an ElevenLabs Professional Voice Clone" + (f" · {eta}" if goals else "")
        print(f"\n{lang}: {s['usable_min']:.1f} usable min of {s['all_min']:.1f} ({s['usable_takes']}/{s['takes']} takes)"
              f" · ~{s['per_day_min']:.0f} usable min/day\n    {eta}")
        print(f"    median: speech {s['speech_dbfs']:.0f} dBFS · room {s['floor_dbfs']:.0f} dBFS · SNR {s['snr_db']:.0f} dB")
        if s["reasons"]:
            print("    not usable: " + ", ".join(f"{k} ×{v}" for k, v in s["reasons"].items()))
    print()
    for line in vb.verdict(st):
        print(line)


def cmd_quit(args, cfg: Config) -> None:
    """Quit the running app cleanly (it will NOT be auto-restarted; reopen it to start again)."""
    import signal
    import subprocess

    out = subprocess.run(["pgrep", "-f", "DictDact.app/Contents/MacOS/python -m dictdact run"],
                         capture_output=True, text=True).stdout.split()
    if not out:
        print("DictDact is not running")
        return
    for pid in out:
        os.kill(int(pid), signal.SIGUSR1)
    print(f"asked DictDact (pid {', '.join(out)}) to quit")


def _app_pids() -> list[str]:
    import subprocess

    return subprocess.run(["pgrep", "-f", "DictDact.app/Contents/MacOS/python -m dictdact run"],
                          capture_output=True, text=True).stdout.split()


def cmd_open(args, cfg: Config) -> None:
    """Bring up the DictDact window (same as Ctrl+Alt+H)."""
    import signal

    pids = _app_pids()
    if not pids:
        print("DictDact is not running")
        return
    os.kill(int(pids[0]), signal.SIGUSR2)


def cmd_ui(args, cfg: Config) -> None:
    """Open the running app's dashboard in the default browser."""
    import subprocess

    from .config import DATA_DIR

    p = DATA_DIR / "ui-url"
    if not p.exists():
        print("DictDact is not running (no ui-url)")
        return
    url = p.read_text().strip()
    if args.print:
        print(url)
    else:
        subprocess.Popen(["open", url])


def cmd_setup(args, cfg: Config) -> None:
    from huggingface_hub import snapshot_download

    for repo in (cfg.whisper_model, cfg.mlx_formatter_model, cfg.parakeet_model):
        print(f"downloading {repo} …", flush=True)
        snapshot_download(repo)
    print("done")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dictdact", description="Local push-to-talk dictation (Spanish + English).")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start the menu-bar app (default)")
    r.add_argument("--headless", action="store_true", help="no menu bar; print to terminal")
    r.add_argument("--mic", help="input device name (substring ok) for this session")
    r.add_argument("--engine", choices=["whisper", "parakeet"])
    r.add_argument("--formatter", choices=["mlx", "ollama", "rules", "off"])
    r.add_argument("--no-paste", action="store_true")

    sub.add_parser("devices", help="list input devices")
    t = sub.add_parser("transcribe", help="run the pipeline on audio file(s)")
    t.add_argument("files", nargs="+")
    t.add_argument("--paste", action="store_true", help="also deliver the result like a live dictation")
    t.add_argument("--engine", choices=["whisper", "parakeet"])
    t.add_argument("--formatter", choices=["mlx", "ollama", "rules", "off"])
    h = sub.add_parser("history", help="show recent dictation blocks")
    h.add_argument("-n", type=int, default=10)
    sub.add_parser("doctor", help="check permissions, devices and models")
    c = sub.add_parser("config", help="show/update config")
    c.add_argument("--mic", help="microphone name, or 'default'")
    c.add_argument("--engine", choices=["whisper", "parakeet"])
    c.add_argument("--formatter", choices=["mlx", "ollama", "rules", "off"])
    c.add_argument("--hotkey", choices=["ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_r", "cmd_r", "shift_r"])
    c.add_argument("--language", help="'auto' (Spanish + English), 'es' or 'en' (locked)")
    c.add_argument("--set", action="append", metavar="KEY=VALUE", help="set any config key")
    v = sub.add_parser("vocab", help="the dictionary: names & tools spelled exactly the way you want")
    vs = v.add_subparsers(dest="vcmd")
    vs.add_parser("list", help="show the dictionary (default)")
    va = vs.add_parser("add", help='add a term: dictdact vocab add "Acme Labs" --spoken "ack me labs, acne labs"')
    va.add_argument("term")
    va.add_argument("--spoken", help="comma-separated ways the speech model tends to write it")
    va.add_argument("--note", help="short hint for the formatter, e.g. 'the company'")
    va.add_argument("--fuzzy", action="store_true", help="also catch sound-alike spellings of this name (Hirox → Hyrox)")
    va.add_argument("--no-fuzzy", action="store_true", help="switch sound-alike matching off for an existing term")
    vr = vs.add_parser("remove", help="remove a term")
    vr.add_argument("term")
    vt = vs.add_parser("test", help="apply the dictionary to some text and print the result")
    vt.add_argument("text", nargs="+")
    vg = vs.add_parser("suggest", help="names your own dictations spell more than one way — candidates to add")
    vg.add_argument("-n", type=int, default=20)
    b = sub.add_parser("voicebank", help="collect your own voice from dictation for a voice clone (ElevenLabs / local)")
    b.add_argument("--min-snr", type=float, default=20.0, help="dB of speech above the room for a take to count")
    bs = b.add_subparsers(dest="bcmd")
    bs.add_parser("status", help="minutes collected per language + a verdict on the mic (default)")
    bs.add_parser("on", help="start keeping your takes (live, no restart)")
    bs.add_parser("off", help="stop keeping takes")
    be = bs.add_parser("export", help="write ElevenLabs-ready files from the cleanest takes")
    be.add_argument("--lang", choices=["es", "en"], required=True)
    be.add_argument("--minutes", type=float, default=180.0, help="how much to export (ElevenLabs: 30 min minimum, 2–3 h best)")
    be.add_argument("--format", choices=["mp3", "wav"], default="mp3", help="mp3 = 192 kbps (what ElevenLabs recommends)")
    be.add_argument("--chunk-min", type=float, default=10.0, help="minutes per file")
    be.add_argument("--out", type=Path, help="folder (default: voice-bank/exports/<lang>-<time>)")
    be.add_argument("--open", action="store_true", help="reveal the folder in Finder")
    br = bs.add_parser("reference", help="cut a ~20 s reference clip for zero-shot cloning")
    br.add_argument("--lang", choices=["es", "en"], required=True)
    br.add_argument("--seconds", type=float, default=20.0)
    br.add_argument("--out", type=Path)
    bsay = bs.add_parser("say", help="speak text in your voice with the local clone (Chatterbox, no cloud)")
    bsay.add_argument("text", nargs="+")
    bsay.add_argument("--lang", choices=["es", "en"], default="es")
    bsay.add_argument("--out", type=Path)
    sub.add_parser("setup", help="download the models")
    sub.add_parser("quit", help="quit the running app cleanly (no auto-restart)")
    sub.add_parser("open", help="show the DictDact window (same as Ctrl+Alt+H)")
    u = sub.add_parser("ui", help="open the dashboard in your browser")
    u.add_argument("--print", action="store_true", help="print the URL instead of opening it")

    args = p.parse_args(argv)
    setup_logging(args.verbose)
    cfg = Config.load()
    # per-session overrides
    for key in ("mic", "engine", "formatter"):
        val = getattr(args, key, None)
        if val:
            setattr(cfg, "microphone" if key == "mic" else key, val)
    if getattr(args, "no_paste", False):
        cfg.paste = False

    cmd = args.cmd or "run"
    if cmd == "run" and args.cmd is None:
        args.headless = False
    {
        "run": cmd_run, "devices": cmd_devices, "transcribe": cmd_transcribe, "history": cmd_history,
        "doctor": cmd_doctor, "config": cmd_config, "setup": cmd_setup, "quit": cmd_quit, "ui": cmd_ui, "open": cmd_open,
        "vocab": cmd_vocab, "voicebank": cmd_voicebank,
    }[cmd](args, cfg)


if __name__ == "__main__":
    main()
