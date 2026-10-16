import signal

from dictdact.instance import RUN, STAND_DOWN, YIELD, ExitGuard, claim_single_instance


def test_no_instances_runs():
    assert claim_single_instance(poll=lambda: [], managed=False) == RUN


def test_unmanaged_copy_yields_to_anything_running():
    assert claim_single_instance(poll=lambda: [111], managed=False) == YIELD


def test_managed_instance_evicts_others(monkeypatch):
    import dictdact.instance as inst
    t = [0.0]
    monkeypatch.setattr(inst.time, "monotonic", lambda: t[0])
    alive = {111: True, 222: True}
    sent = []

    def kill(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGTERM and pid == 111:
            alive[111] = False          # 111 dies on SIGTERM, 222 ignores it

    def sleep(s):
        t[0] += s

    ok = claim_single_instance(grace_s=3, poll=lambda: [p for p, a in alive.items() if a],
                               managed=True, kill=kill, sleep=sleep)
    assert ok == RUN
    assert (111, signal.SIGTERM) in sent and (222, signal.SIGTERM) in sent
    assert (222, signal.SIGKILL) in sent and (111, signal.SIGKILL) not in sent


def test_exit_guard_user_quit_does_not_exit(monkeypatch):
    import dictdact.instance as inst
    calls = []
    monkeypatch.setattr(inst.os, "_exit", lambda code: calls.append(code))
    g = ExitGuard()
    g.user_quit = True
    g.on_terminate()
    assert calls == []
    g.user_quit = False
    g.on_terminate()
    assert calls == [1]


def test_launchd_labels_and_managed_detection(monkeypatch):
    import dictdact.instance as inst
    monkeypatch.setattr(inst, "bundle_id", lambda: "com.x.dictdact")
    assert inst.launchd_labels() == ["com.x.dictdact", "com.x.dictdact.agent"]

    class R:
        def __init__(self, stdout): self.stdout = stdout
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: R(f"\tstate = running\n\tpid = {inst.os.getpid()}\n"))
    assert inst.is_launchd_managed() is True
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: R("\tpid = 1\n"))
    monkeypatch.setenv("XPC_SERVICE_NAME", "application.com.x.dictdact.1.2")
    assert inst.is_launchd_managed() is False
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.x.dictdact")
    assert inst.is_launchd_managed() is True


def test_lower_ranked_job_stands_down_to_legacy(monkeypatch):
    import dictdact.instance as inst
    disabled = []
    monkeypatch.setattr(inst, "disable_job", lambda label: disabled.append(label) or True)
    monkeypatch.setattr(inst, "bundle_id", lambda: "com.x.dictdact")
    # we are the .agent job; the legacy job (pid 100) is running
    ok = claim_single_instance(poll=lambda: [100], managed=True, mine="com.x.dictdact.agent",
                               jobs={"com.x.dictdact": 100, "com.x.dictdact.agent": inst.os.getpid()},
                               kill=lambda p, s: None, sleep=lambda s: None)
    assert ok == STAND_DOWN and disabled == ["com.x.dictdact.agent"]


def test_legacy_job_disables_lower_ranked_agent_and_evicts(monkeypatch):
    import dictdact.instance as inst
    disabled, sent = [], []
    monkeypatch.setattr(inst, "disable_job", lambda label: disabled.append(label) or True)
    monkeypatch.setattr(inst, "bundle_id", lambda: "com.x.dictdact")
    alive = {200: True}

    def kill(pid, sig):
        sent.append((pid, sig)); alive[pid] = False
    ok = claim_single_instance(poll=lambda: [p for p, a in alive.items() if a], managed=True, mine="com.x.dictdact",
                               jobs={"com.x.dictdact": inst.os.getpid(), "com.x.dictdact.agent": 200},
                               kill=kill, sleep=lambda s: None)
    assert ok == RUN and disabled == ["com.x.dictdact.agent"] and sent and sent[0][0] == 200


def test_disable_failure_never_loops(monkeypatch):
    import dictdact.instance as inst
    monkeypatch.setattr(inst, "disable_job", lambda label: False)
    monkeypatch.setattr(inst, "bundle_id", lambda: "com.x.dictdact")
    # loser cannot disable itself → still stands down with a clean exit (no KeepAlive bounce)
    assert claim_single_instance(poll=lambda: [100], managed=True, mine="com.x.dictdact.agent",
                                 jobs={"com.x.dictdact": 100, "com.x.dictdact.agent": inst.os.getpid()},
                                 kill=lambda p, s: None, sleep=lambda s: None) == STAND_DOWN
    # winner cannot disable the lower job → must NOT kill it (launchd would bounce it); runs anyway
    sent = []
    assert claim_single_instance(poll=lambda: [200], managed=True, mine="com.x.dictdact",
                                 jobs={"com.x.dictdact": inst.os.getpid(), "com.x.dictdact.agent": 200},
                                 kill=lambda p, s: sent.append(p), sleep=lambda s: None) == RUN
    assert sent == []


def test_offscreen_geometry_math():
    """The rule the window uses: recenter when less than half of it overlaps a live screen."""
    def visible_fraction(win, screens):
        area = win[2] * win[3]
        best = 0.0
        for s in screens:
            w = max(0.0, min(win[0] + win[2], s[0] + s[2]) - max(win[0], s[0]))
            h = max(0.0, min(win[1] + win[3], s[1] + s[3]) - max(win[1], s[1]))
            best = max(best, w * h)
        return best / area

    laptop = [(0, 0, 1512, 950)]
    assert visible_fraction((100, 100, 900, 600), laptop) == 1.0          # on screen
    assert visible_fraction((3000, 200, 900, 600), laptop) == 0.0         # on a disconnected display
    assert visible_fraction((1300, 100, 900, 600), laptop) < 0.5          # mostly hanging off
