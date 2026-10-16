"""The re-paste shortcut is configurable.

It has its own key on purpose: ⌘V pastes whatever the clipboard happens to hold, this always
pastes your last dictation. Any letter can be chosen (e.g. ctrl+alt+s instead of ctrl+alt+v)."""
from __future__ import annotations

from pynput.keyboard import Key, KeyCode

from dictdact.config import Config
from dictdact.hotkey import DEFAULT_RETRIEVE_COMBO, Hotkeys, parse_combo, pretty_combo

import pytest


def _keys(spec: str):
    """The physical presses for a combo, modifiers first, as pynput reports them."""
    mods, letter = parse_combo(spec)
    order = [("ctrl", Key.ctrl), ("alt", Key.alt), ("shift", Key.shift), ("cmd", Key.cmd)]
    from dictdact.hotkey import _VK
    return [k for name, k in order if name in mods] + [KeyCode.from_vk(_VK[letter])]


def _hotkeys(spec: str, fired: list):
    return Hotkeys("ctrl", 180, True, on_start=lambda: None, on_stop=lambda: None,
                   on_cancel=lambda: None, on_lock=lambda: None,
                   on_retrieve=lambda: fired.append(spec), retrieve_combo=spec)


@pytest.mark.parametrize("spec", ["ctrl+alt+s", "ctrl+alt+v", "cmd+shift+d", "alt+shift+r"])
def test_the_configured_combo_is_what_fires(spec):
    fired: list = []
    hk = _hotkeys(spec, fired)
    for k in _keys(spec):
        hk._on_press(k)
    assert fired == [spec]


def test_the_old_shortcut_stops_firing_once_it_is_changed():
    """Whoever re-binds it must not be left with two live shortcuts."""
    fired: list = []
    hk = _hotkeys("ctrl+alt+s", fired)
    for k in _keys("ctrl+alt+v"):
        hk._on_press(k)
    assert fired == []


def test_the_letter_is_matched_by_key_code_not_character():
    """⌥+s types 'ß' on a Spanish layout: the character is useless, the virtual key is not."""
    fired: list = []
    hk = _hotkeys("ctrl+alt+s", fired)
    hk._on_press(Key.ctrl)
    hk._on_press(Key.alt)
    hk._on_press(KeyCode.from_vk(1))          # ANSI 's', whatever character the layout produced
    assert fired == ["ctrl+alt+s"]


def test_a_shortcut_that_cannot_work_is_rejected_not_silently_ignored():
    with pytest.raises(ValueError):
        parse_combo("s")                       # no modifier: would fire while the user types
    with pytest.raises(ValueError):
        parse_combo("ctrl+alt+1")              # not in the virtual-key table
    with pytest.raises(ValueError):
        parse_combo("ctrl+alt")                # no letter at all


def test_pretty_combo_has_one_spelling():
    assert pretty_combo("ctrl+alt+s") == "Ctrl+Alt+S"
    assert pretty_combo("alt+ctrl+s") == "Ctrl+Alt+S"     # order in the file never leaks to the menu
    assert pretty_combo("cmd+shift+d") == "Shift+Cmd+D"


def test_the_default_is_unchanged_for_everyone_else():
    assert Config().retrieve_combo == DEFAULT_RETRIEVE_COMBO == "ctrl+alt+v"
