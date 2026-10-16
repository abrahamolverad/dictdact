"""Unit tests for dictdact.format (self-correction handling, prompts, formatter/vocab
integration) and dictdact.output.wait_modifiers_released.

No ML models: FakeLLM below subclasses _LLMFormatter without ever touching MLXFormatter /
OllamaFormatter, and wait_modifiers_released is exercised directly (no simulator/device).
"""
import time

import pytest

from dictdact.format import (STYLES, SYSTEM_PROMPT, _STOPWORDS, NoFormatter, RulesFormatter, _LLMFormatter,
                             apply_spoken_commands, needs_llm, style_prompt, system_prompt, words_kept)
from dictdact.output import wait_modifiers_released
from dictdact.vocab import Term, Vocabulary


# -- apply_spoken_commands: etcétera -------------------------------------------------------

def test_etcetera_variants_become_etc_dot():
    out = apply_spoken_commands("los filtros de precio, categoría, etcétera y más")
    assert "etc." in out
    assert "etcétera" not in out


def test_etcetera_ascii_and_trailing_dot_variants():
    assert apply_spoken_commands("et cetera") == "etc."
    assert apply_spoken_commands("etcetera.") == "etc."   # no double dot


# -- needs_llm --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "mándalo el lunes, no, quiero decir el martes",
    "send it to John, I mean Daniel",
    "ship Friday, actually Thursday",
    "scratch that",
    "borra eso",
    "y etcétera",
])
def test_needs_llm_true_for_self_corrections_and_markers(raw):
    assert needs_llm(raw)


def test_needs_llm_false_for_a_short_clean_take():
    assert not needs_llm("hola mundo")


# -- every correction marker the prompt tells the model to drop is verbatim-exempt --------

@pytest.mark.parametrize("marker", [
    "quiero", "decir", "actually", "scratch", "rephrase", "correction", "borra", "tacha",
    "olvida", "eso", "etcetera", "etc", "digo", "perdon", "mejor", "dicho", "wait", "espera",
    "sorry",
])
def test_correction_markers_are_stopwords(marker):
    assert marker in _STOPWORDS


# -- words_kept does not punish a perfect self-correction cleanup -------------------------

def test_words_kept_self_correction_cleanup_scores_high():
    pre = ("quiero que mandes el reporte financiero completo del trimestre a la oficina "
           "central el lunes, no, quiero decir el martes, junto con las facturas pendientes "
           "de pago")
    out = ("quiero que mandes el reporte financiero completo del trimestre a la oficina "
           "central el martes, junto con las facturas pendientes de pago")
    assert words_kept(pre, out) >= 0.60


# -- system_prompt / style_prompt ----------------------------------------------------------

def test_system_prompt_default_keeps_the_mix():
    assert "keep the mix exactly" in system_prompt()


def test_system_prompt_locked_spanish_drops_the_mix_rule():
    p = system_prompt(language="es")
    assert "locked the dictation language to Spanish" in p
    assert "keep the mix exactly" not in p


def test_system_prompt_locked_english_says_english():
    assert "English" in system_prompt(language="en")


def test_system_prompt_glossary_included():
    p = system_prompt(glossary="Acme Labs (the company)")
    assert "Glossary" in p
    assert "Acme Labs (the company)" in p


def test_system_prompt_style_summary_uses_style_prompt():
    p = system_prompt(style="summary", glossary="X")
    assert "Spell the speaker's own names" in p
    assert "X" in p


def test_system_prompt_constant_and_style_prompt_alias():
    assert SYSTEM_PROMPT == system_prompt()
    assert style_prompt("clean") == system_prompt()


# -- formatter + vocabulary integration ----------------------------------------------------

class FakeLLM(_LLMFormatter):
    name = "fake"
    always = True

    def __init__(self, reply):
        self.reply = reply
        self.captured_raw = None

    def _complete(self, raw):
        self.captured_raw = raw
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


RAW = "let's work on ack me labs project"


def _vocab():
    return Vocabulary([Term("Acme Labs", ["ack me labs"])])


def test_llm_formatter_sends_vocabulary_applied_text_to_complete():
    f = FakeLLM("Let's work on Acme Labs project.")
    f.vocab = _vocab()
    f.format(RAW)
    assert "Acme Labs" in f.captured_raw
    assert "ack me labs" not in f.captured_raw


def test_llm_formatter_fixes_a_reply_that_re_mangles_the_term():
    f = FakeLLM("Let's work on ack me labs project.")   # LLM re-introduces the raw spelling
    f.vocab = _vocab()
    out = f.format(RAW)
    assert out == "Let's work on Acme Labs project."


def test_llm_formatter_fallback_path_also_applies_vocabulary():
    f = FakeLLM(RuntimeError("boom"))
    f.vocab = _vocab()
    out = f.format(RAW)
    assert "Acme Labs" in out
    assert "ack me labs" not in out
    assert f.last_fallback == "the model failed"


def test_rules_formatter_applies_vocabulary():
    f = RulesFormatter()
    f.vocab = _vocab()
    assert f.format("ack me labs plan") == "Acme Labs plan"


def test_no_formatter_applies_vocabulary():
    f = NoFormatter()
    f.vocab = _vocab()
    assert f.format("ack me labs plan") == "Acme Labs plan"


def test_glossary_property_reflects_vocab_or_is_empty():
    f = NoFormatter()
    f.vocab = _vocab()
    assert f.glossary == _vocab().glossary()
    g = NoFormatter()
    assert g.vocab is None
    assert g.glossary == ""


# -- dictdact.output.wait_modifiers_released -----------------------------------------------

def test_wait_modifiers_released_returns_bool_quickly():
    t0 = time.monotonic()
    result = wait_modifiers_released(timeout_s=0.05)
    elapsed = time.monotonic() - t0
    assert isinstance(result, bool)
    assert elapsed < 0.5


# -- guards must excuse what a self-correction abandons ---------------------------------

def test_abandonable_covers_marker_and_words_before_it():
    from dictdact.format import abandonable
    words, chars = abandonable("mándalo el lunes, no, quiero decir el martes por la tarde")
    assert "lunes" in words and "mandalo" in words
    assert "martes" not in words and "tarde" not in words
    assert chars > 0
    assert abandonable("hola mundo sin correcciones") == (set(), 0)


def test_long_self_correction_passes_both_guards():
    from dictdact.format import output_is_sane, words_kept
    pre = ("prepara el reporte de ventas de la región norte con las gráficas de cada sucursal y el "
           "comparativo anual, scratch that, solo mándame el total del mes")
    out = "Solo mándame el total del mes."
    assert len(out) < 0.35 * len(pre)            # the old length guard alone would reject this
    assert output_is_sane(pre, out)
    assert words_kept(pre, out) >= 0.60


def test_summary_without_a_correction_is_still_rejected():
    from dictdact.format import output_is_sane, words_kept
    pre = ("prepara el reporte de ventas de la región norte con las gráficas de cada sucursal y el "
           "comparativo anual y el desglose por producto y por vendedor para la junta del lunes")
    out = "Reporte de ventas para la junta."
    assert not output_is_sane(pre, out)
    assert words_kept(pre, out) < 0.60


def test_paste_is_refused_while_modifiers_stay_held(monkeypatch):
    import dictdact.output as o
    import dictdact.perms as perms
    board = {}
    sent = []
    monkeypatch.setattr(perms, "accessibility", lambda: True)
    monkeypatch.setattr(o, "get_clipboard_text", lambda: board.get("text"))
    monkeypatch.setattr(o, "set_clipboard_text", lambda t: board.__setitem__("text", t))
    monkeypatch.setattr(o, "_send_cmd_v", lambda: sent.append("⌘V"))
    monkeypatch.setattr(o, "wait_modifiers_released", lambda timeout_s=0: False)
    assert o.paste_text("hola", restore_clipboard=True) is False
    assert sent == [] and board["text"] == "hola"        # nothing sent; the text is in the clipboard
    monkeypatch.setattr(o, "wait_modifiers_released", lambda timeout_s=0: True)
    monkeypatch.setattr(o.time, "sleep", lambda s: None)
    assert o.paste_text("adiós", restore_clipboard=False) is True
    assert sent == ["⌘V"] and board["text"] == "adiós"


# -- the exemption must not be a hole in the verbatim guard -------------------------------

def test_plain_adverb_actually_excuses_nothing():
    from dictdact.format import abandonable
    assert abandonable("it actually works reliably for every customer today and we ship it") == (set(), 0)
    assert abandonable("Actually, the report is ready and the customer signed it") == (set(), 0)
    words, _ = abandonable("ship it on Friday, actually on Thursday morning")
    assert "friday" in words and "thursday" not in words


def test_summary_with_an_adverbial_actually_is_still_rejected():
    from dictdact.format import output_is_sane, words_kept
    pre = ("so the onboarding flow actually works reliably now for every customer we tested today, "
           "the billing page still shows the old logo, and the export button crashes on large files")
    out = "It works reliably for every customer today."
    assert not output_is_sane(pre, out) or words_kept(pre, out) < 0.60


def test_capitalized_delete_marker_is_excused():
    from dictdact.format import abandonable, output_is_sane, words_kept
    pre = ("Prepara el reporte de ventas de la región norte con las gráficas de cada sucursal y el "
           "comparativo anual. Scratch that. Solo mándame el total del mes")
    out = "Solo mándame el total del mes."
    words, _ = abandonable(pre)
    assert "reporte" in words and "sucursal" in words
    assert output_is_sane(pre, out) and words_kept(pre, out) >= 0.60
    pre_es = "Agrega los filtros de precio y categoría y el buscador y el ordenamiento, Borra eso, solo el buscador"
    assert "precio" in abandonable(pre_es)[0]


def test_replace_windows_cannot_excuse_most_of_a_take():
    from dictdact.format import abandonable
    # three replace markers over a long take would excuse almost everything — refused
    pre = ("mándame el reporte mensual completo, digo el resumen ejecutivo de ventas, quiero decir el "
           "total consolidado de la región norte, no, mejor el comparativo anual de la sucursal principal")
    words, _ = abandonable(pre)
    assert words == set()
    # a short take keeps its (small) exemption: the guards have no signal there anyway
    assert "lunes" in abandonable("mándalo el lunes, no, quiero decir el martes")[0]


# -- ordinary language must not look like a correction ------------------------------------

def test_instruction_shaped_phrases_are_not_markers():
    from dictdact.format import abandonable
    assert abandonable("open the workbook, check the formula in the audit column and scratch that row") == (set(), 0)
    assert abandonable("borra eso del archivo y pon el nuevo logo") == (set(), 0)
    assert abandonable("no quiero decir que el reporte esté mal, solo que llegó tarde") == (set(), 0)
    assert abandonable("send the totals, I mean that the board wants them by Friday") == (set(), 0)
    # the genuine forms still count
    assert "workbook" in abandonable("open the workbook and check the formula, scratch that, just send the totals")[0]
    assert "lunes" in abandonable("mándalo el lunes, no, quiero decir el martes")[0]


def test_summary_after_an_instruction_shaped_scratch_that_is_rejected():
    from dictdact.format import output_is_sane, words_kept
    pre = ("open the workbook, verify every formula in the audit column, keep the totals visible, "
           "then scratch that row and export the sheet for the accountant before the meeting")
    out = "Keep the totals visible."
    assert not output_is_sane(pre, out) or words_kept(pre, out) < 0.60


# -- apologies and kept markers are ordinary language ------------------------------------

def test_sorry_and_correction_are_ordinary_language():
    from dictdact.format import abandonable
    assert abandonable("thanks for the update, sorry for the delay, the invoice goes out today") == (set(), 0)
    assert abandonable("send the totals and the audit trail, correction. Then export the sheet") == (set(), 0)


def test_a_marker_the_model_kept_excuses_nothing():
    from dictdact.format import abandonable
    pre = "send it to John, I mean to Daniel, tonight"
    assert "john" in abandonable(pre, "Send it to Daniel tonight.")[0]
    assert abandonable(pre, "Send it to John, I mean to Daniel, tonight.") == (set(), 0)


# -- explanatory adverbs and whole-word markers -------------------------------------------

def test_explanatory_adverb_after_a_marker_is_not_a_correction():
    from dictdact.format import abandonable
    assert abandonable("send the proposal with the ten details, I mean honestly the client wants them all") == (set(), 0)
    assert abandonable("manda la propuesta con los diez detalles, digo sinceramente el cliente los quiere") == (set(), 0)
    assert "john" in abandonable("send it to John, I mean Daniel", "Send it to Daniel.")[0]


def test_marker_consumption_is_whole_word():
    from dictdact.format import abandonable, words_kept
    pre = "envía el reporte completo con las gráficas, digo envía solo el código mañana"
    out = "Envía solo el código mañana."
    assert "reporte" in abandonable(pre, out)[0]          # "código" does not hide "digo"
    assert words_kept(pre, out) >= 0.60 or len(set(pre.split())) < 12


# -- parenthetical markers ------------------------------------------------------------------

def test_parenthetical_marker_is_not_a_correction():
    from dictdact.format import abandonable
    assert abandonable("send the proposal with the ten details, I mean, the client wants them all") == (set(), 0)
    assert abandonable("send the proposal with the ten details, I mean — personally I would keep them") == (set(), 0)
    assert abandonable("manda la propuesta con los diez detalles, digo, francamente el cliente los quiere") == (set(), 0)
    assert "john" in abandonable("send it to John, I mean Daniel", "Send it to Daniel.")[0]
