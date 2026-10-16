"""The formatting pass has a wall-clock ceiling.

mlx_lm.generate() runs until the model is done. When another app saturates the GPU (a 28 GB
model parked in Ollama, a ComfyUI render), takes that format in ~3 s on an idle machine took
22-45 s — measured on 767 real takes, the median throughput fell from 411 chars/s to 52. The
user cannot tell a busy machine from a broken app; they just wait. Past the budget the pass is
abandoned and the deterministic cleanup delivers their words instead."""
from __future__ import annotations

import time

import pytest

from dictdact.format import FormatterTimeout, MLXFormatter, _LLMFormatter


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "SYS|" + messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 for c in text]


class _SlowLLM(_LLMFormatter):
    name = "slow"
    always = True

    def _complete(self, raw, budget_s=None):
        raise FormatterTimeout("the formatting pass passed its 12s budget")


def test_a_timed_out_pass_falls_back_to_the_speakers_own_words():
    f = _SlowLLM()
    out = f.format("um so the plan is to ship the release on monday and then test everything twice")
    assert out == "So the plan is to ship the release on monday and then test everything twice."
    assert f.last_fallback == "the formatter ran out of time"   # the pipeline surfaces this


def _mlx(monkeypatch, per_token_s: float, tokens=("one ", "two ", "three ", "four ")):
    """An MLXFormatter whose generation yields `tokens`, `per_token_s` apart."""
    import mlx_lm

    class _Resp:
        def __init__(self, text):
            self.text = text

    def fake_stream(model, tok, prompt, max_tokens=256, **kw):
        for t in tokens:
            time.sleep(per_token_s)
            yield _Resp(t)

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream)
    f = MLXFormatter("fake/model")
    f._model, f._tok = object(), _Tok()
    return f


def test_generation_stops_at_the_deadline(monkeypatch):
    f = _mlx(monkeypatch, per_token_s=0.05)
    f.timeout_s = 0.12
    t0 = time.monotonic()
    with pytest.raises(FormatterTimeout):
        f._complete("a take long enough to matter")
    assert time.monotonic() - t0 < 0.5      # abandoned, not run to the end


def test_a_zero_budget_never_interrupts(monkeypatch):
    f = _mlx(monkeypatch, per_token_s=0.01)
    f.timeout_s = 0.0
    assert f._complete("hola") == "one two three four "
    # keep-warm touches and the warm-up pass 0 explicitly: a slow touch is the symptom of a
    # busy GPU, not a failure, and interrupting it would leave the weights half-paged.
    f.timeout_s = 0.001
    assert f._complete("hola", budget_s=0) == "one two three four "


def test_an_abandoned_pass_leaves_the_shared_prefix_cache_clean(monkeypatch):
    """The KV cache is shared by every take on the model worker. A pass that stopped early
    must not leave its tokens there for the next dictation to inherit."""
    import mlx_lm.models.cache as cache_mod

    f = _mlx(monkeypatch, per_token_s=0.05)
    f.timeout_s = 0.12

    class _Slot:
        offset = 0

    trimmed: list[int] = []
    monkeypatch.setattr(cache_mod, "trim_prompt_cache", lambda c, n: trimmed.append(n))
    f._cache, f._prefix_ids = [_Slot()], []
    _Slot.offset = 7                       # the abandoned pass advanced the cache

    with pytest.raises(FormatterTimeout):
        f._complete("a take long enough to matter")
    assert trimmed == [7]


def test_clipboard_restore_is_off_by_default():
    """Nothing can confirm a ⌘V landed. Restoring the previous clipboard 0.6 s later took
    the dictation away from the one retry that always works: pressing ⌘V again."""
    from dictdact.config import Config

    assert Config().restore_clipboard is False
    assert Config().formatter_timeout_s == 12.0
