"""Start at login + restart on crash.

Two flavours, detected automatically:
* the release bundle ships a LaunchAgent plist inside the app and registers it via SMAppService;
* a developer install (scripts/install.sh) uses a legacy plist in ~/Library/LaunchAgents, driven by launchctl.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("dictdact.autostart")

PLIST_NAME = "com.abrahamolvera.dictdact.agent.plist"
LEGACY_LABEL = "com.abrahamolvera.dictdact"
LEGACY_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LEGACY_LABEL}.plist"


def _bundle_plist() -> Path | None:
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.name == "Contents":
            p = parent / "Library" / "LaunchAgents" / PLIST_NAME
            return p if p.exists() else None
    return None


def flavour() -> str | None:
    if _bundle_plist() is not None:
        try:
            from ServiceManagement import SMAppService  # noqa: F401

            return "bundle"
        except Exception:
            return None
    if LEGACY_PLIST.exists():
        return "legacy"
    return None


def available() -> bool:
    return flavour() is not None


def _service():
    from ServiceManagement import SMAppService

    return SMAppService.agentServiceWithPlistName_(PLIST_NAME)


def _legacy_target() -> str:
    return f"gui/{os.getuid()}/{LEGACY_LABEL}"


def _legacy_loaded() -> bool:
    r = subprocess.run(["launchctl", "print", _legacy_target()], capture_output=True, text=True, timeout=5)
    return r.returncode == 0


def enabled() -> bool:
    try:
        f = flavour()
        if f == "bundle":
            return int(_service().status()) == 1  # SMAppServiceStatusEnabled
        if f == "legacy":
            return _legacy_loaded()
    except Exception:
        pass
    return False


def requires_approval() -> bool:
    try:
        return flavour() == "bundle" and int(_service().status()) == 2
    except Exception:
        return False


def set_enabled(on: bool) -> bool:
    try:
        f = flavour()
        if f == "bundle":
            svc = _service()
            ok, err = (svc.registerAndReturnError_(None) if on else svc.unregisterAndReturnError_(None))
            if not ok:
                log.warning("autostart %s failed: %s", "register" if on else "unregister", err)
            return bool(ok)
        if f == "legacy":
            uid = os.getuid()
            if on:
                subprocess.run(["launchctl", "enable", _legacy_target()], capture_output=True, timeout=10)
                r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(LEGACY_PLIST)], capture_output=True, text=True, timeout=10)
                return r.returncode == 0 or _legacy_loaded()
            # Off: the job must not relaunch at login, but the running app keeps running (bootout
            # would kill us). launchctl disable does exactly that for the next login.
            r = subprocess.run(["launchctl", "disable", _legacy_target()], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
    except Exception as e:
        log.warning("autostart unavailable: %s", e)
    return False
