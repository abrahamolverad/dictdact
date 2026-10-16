"""macOS permission checks (Input Monitoring, Accessibility, Microphone)."""
from __future__ import annotations


def input_monitoring(prompt: bool = False) -> bool | None:
    try:
        from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess
    except Exception:
        return None
    if CGPreflightListenEventAccess():
        return True
    if prompt:
        return bool(CGRequestListenEventAccess())
    return False


def accessibility(prompt: bool = False) -> bool | None:
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    except Exception:
        return None
    return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))


def microphone() -> str:
    """'granted' | 'denied' | 'undetermined' | 'unknown' (needs AVFoundation bindings)."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except Exception:
        return "unknown"
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
    return {0: "undetermined", 1: "restricted", 2: "denied", 3: "granted"}.get(int(status), "unknown")


def request_microphone(callback=None) -> str:
    """Trigger the system microphone prompt if the decision is still pending.
    Returns the current status ('granted' | 'denied' | 'undetermined' | 'unknown')."""
    status = microphone()
    if status == "undetermined":
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeAudio, lambda granted: callback(bool(granted)) if callback else None)
        except Exception:
            pass
    return status


def report() -> list[tuple[str, bool | None, str]]:
    return [
        ("Input Monitoring (hold Ctrl to dictate)", input_monitoring(),
         "System Settings → Privacy & Security → Input Monitoring"),
        ("Accessibility (paste with ⌘V)", accessibility(),
         "System Settings → Privacy & Security → Accessibility"),
        (f"Microphone ({microphone()})", microphone() in ("granted", "unknown"),
         "System Settings → Privacy & Security → Microphone"),
    ]
