"""Single-instance guard and 'unintended exit' detection for the installed app."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Callable

log = logging.getLogger("dictdact.instance")


def bundle_id() -> str | None:
    try:
        from Foundation import NSBundle

        bid = NSBundle.mainBundle().bundleIdentifier()
        return str(bid) if bid else None
    except Exception:
        return None


def other_instance_pids(bid: str) -> list[int]:
    from AppKit import NSRunningApplication

    me = os.getpid()
    return [int(a.processIdentifier()) for a in NSRunningApplication.runningApplicationsWithBundleIdentifier_(bid)
            if int(a.processIdentifier()) != me and not a.isTerminated()]


def launchd_labels(bid: str | None = None) -> list[str]:
    bid = bid or bundle_id() or "com.abrahamolvera.dictdact"
    return [bid, bid + ".agent"]


def _job_pid(label: str) -> int | None:
    try:
        out = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{label}"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    m = re.search(r"^\s*pid = (\d+)", out, re.M)
    return int(m.group(1)) if m else None


def job_pids() -> dict[str, int | None]:
    """pid launchd reports for each of our job labels (None = not running / not loaded)."""
    return {label: _job_pid(label) for label in launchd_labels()}


def managed_label() -> str | None:
    """The launchd job label that owns this process, or None if launched by Finder / a shell.

    Definitive check: `launchctl print gui/<uid>/<label>` lists our pid. Fallback: launchd sets
    XPC_SERVICE_NAME to the job label, while Finder / `open` launches get
    'application.<bundle-id>.<n>.<n>'."""
    me = os.getpid()
    for label, pid in job_pids().items():
        if pid == me:
            return label
    name = os.environ.get("XPC_SERVICE_NAME", "")
    return name if name in launchd_labels() else None


def is_launchd_managed() -> bool:
    return managed_label() is not None


def disable_job(label: str) -> bool:
    """Permanently stand a job down so two launchd jobs never fight over the hotkey:
    the legacy plist via launchctl, the bundled .agent via SMAppService. Returns True only
    when launchd confirms the job is gone."""
    try:
        if label.endswith(".agent"):
            from . import autostart

            ok = autostart.set_enabled(False) and not autostart.enabled()
        else:
            uid = os.getuid()
            subprocess.run(["launchctl", "disable", f"gui/{uid}/{label}"], capture_output=True, timeout=10)
            r = subprocess.run(["launchctl", "bootout", f"gui/{uid}/{label}"], capture_output=True, timeout=10)
            ok = r.returncode == 0 or _job_pid(label) is None
        if ok:
            log.warning("disabled launchd job %s", label)
        else:
            log.error("could NOT disable launchd job %s", label)
        return ok
    except Exception as e:
        log.error("could not disable job %s: %s", label, e)
        return False


RUN, YIELD, STAND_DOWN = "run", "yield", "stand_down"


def claim_single_instance(grace_s: float = 10.0, poll: Callable[[], list[int]] | None = None,
                          managed: bool | None = None, kill: Callable[[int, int], None] = os.kill,
                          sleep=time.sleep, mine: str | None = None,
                          jobs: dict[str, int | None] | None = None) -> str:
    """Decide what this process should do: RUN, YIELD (exit non-zero: we are an unmanaged copy
    or a job that could not stand down — launchd may retry), or STAND_DOWN (exit 0: our job
    has been disabled, stay down for good).

    * No other instance → run.
    * We are unmanaged (Finder / shell) and anything else runs → yield (return False).
    * We are a launchd job and the other instance is a *higher-ranked* launchd job (rank =
      order in launchd_labels(): the legacy label outranks the bundled .agent) → disable our own
      job and yield, so two jobs can never evict each other in a loop.
    * Otherwise we own the hotkey: disable any lower-ranked job, evict every other copy
      (SIGTERM, then SIGKILL after ``grace_s``) and run."""
    import signal

    bid = bundle_id()
    if poll is None:
        if not bid:
            return True  # dev run from a terminal: no bundle, no guard
        poll = lambda: other_instance_pids(bid)  # noqa: E731
    others = poll()
    if not others:
        return RUN
    if managed is None:
        mine = managed_label() if mine is None else mine
        managed = mine is not None
    if not managed:
        log.warning("another DictDact is already running (pid %s) — this copy exits", others)
        return YIELD
    labels = launchd_labels(bid)
    if mine is None:
        mine = labels[0]
    jobs = job_pids() if jobs is None else jobs
    my_rank = labels.index(mine) if mine in labels else len(labels)
    for label, pid in jobs.items():
        if pid in others and label in labels and labels.index(label) < my_rank:
            log.warning("launchd job %s (pid %s) outranks ours (%s): standing down", label, pid, mine)
            if disable_job(mine):
                return STAND_DOWN
            # Could not disable ourselves: exit 0 anyway so KeepAlive does not bounce us every 5 s.
            log.error("self-disable failed; exiting cleanly to avoid a restart loop")
            return STAND_DOWN
    evict = list(others)
    for label, pid in jobs.items():
        if pid in others and label in labels and labels.index(label) > my_rank:
            if not disable_job(label):
                # launchd would just restart it; leave it — it stands down by itself on its next check.
                log.error("leaving job %s (pid %s) alone: it could not be disabled", label, pid)
                evict.remove(pid)
    if not evict:
        return RUN
    log.warning("launchd-managed instance (%s) evicting other DictDact copies: %s", mine, evict)
    for pid in evict:
        try:
            kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    base_poll = poll
    poll = lambda: [p for p in base_poll() if p in evict]  # noqa: E731
    deadline = time.monotonic() + grace_s
    while poll() and time.monotonic() < deadline:
        sleep(0.5)
    for pid in poll():
        try:
            kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return RUN


class ExitGuard:
    """Turns any termination that was not requested from our own Quit menu into exit code 1,
    so launchd's KeepAlive (SuccessfulExit=false) brings the app back."""

    def __init__(self):
        self.user_quit = False
        self._observer = None

    def install(self) -> None:
        try:
            from Foundation import NSNotificationCenter, NSObject

            guard = self

            class _DictDactTerminationWatcher(NSObject):
                def willTerminate_(self, note):
                    guard.on_terminate()

            self._observer = _DictDactTerminationWatcher.alloc().init()
            NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
                self._observer, "willTerminate:", "NSApplicationWillTerminateNotification", None)
        except Exception as e:  # pragma: no cover - AppKit missing
            log.warning("exit guard not installed: %s", e)

    def on_terminate(self) -> None:
        if self.user_quit:
            log.info("quit from the menu — not restarting")
            return
        log.warning("terminated from outside (not via the Quit menu) — exiting 1 so launchd restarts DictDact")
        for h in logging.getLogger("dictdact").handlers:
            try:
                h.flush()
            except Exception:
                pass
        os._exit(1)
