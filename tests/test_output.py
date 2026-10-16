from unittest import mock

import dictdact.output as out


def test_cmd_v_uses_virtual_keycodes_not_layout_lookup():
    posted = []
    with mock.patch("Quartz.CGEventCreateKeyboardEvent", side_effect=lambda src, vk, down: (vk, down)), \
         mock.patch("Quartz.CGEventSetFlags", side_effect=lambda ev, flags: posted.append((ev, flags))), \
         mock.patch("Quartz.CGEventPost"):
        out._send_cmd_v()
    vks = [ev[0] for ev, _ in posted]
    assert vks == [55, 9, 9, 55]              # cmd down, v down, v up, cmd up
    assert [ev[1] for ev, _ in posted] == [True, True, False, False]
    assert posted[1][1] != 0 and posted[3][1] == 0   # ⌘ flag on V, cleared on cmd-up
