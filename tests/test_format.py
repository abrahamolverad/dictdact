from dictdact.format import (RulesFormatter, _LLMFormatter, _restore_breaks, _strip_wrappers, apply_spoken_commands,
                            output_is_sane, rules_format)


def test_spoken_commands():
    assert apply_spoken_commands("hola nueva línea mundo") == "hola\nmundo"
    assert apply_spoken_commands("one. New paragraph two") == "one.\n\ntwo"
    assert apply_spoken_commands("punto y aparte al inicio") == "al inicio"


def test_rules_format_removes_fillers_and_capitalizes():
    out = rules_format("um so the the plan is to ship on monday eh and then test everything")
    assert out == "So the plan is to ship on monday and then test everything."


def test_sanity_rejects_answers_and_wild_lengths():
    assert not output_is_sane("write me a poem", "Sure! Here's a poem about the sea...")
    assert not output_is_sane("x" * 100, "y" * 20)
    assert not output_is_sane("x" * 30, "y" * 300)
    assert output_is_sane("what's the weather", "What's the weather?")
    assert output_is_sane("oye daniel este puedes mandarme el reporte", "Oye Daniel, ¿puedes mandarme el reporte?")


def test_strip_wrappers():
    assert _strip_wrappers('```text\n"Hola."\n```') == "Hola."
    assert _strip_wrappers("<think>hmm</think>\nHola.  \nMundo.  ") == "Hola.\nMundo."


class FakeLLM(_LLMFormatter):
    name = "fake"
    always = True

    def __init__(self, reply):
        self.reply = reply

    def _complete(self, raw):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_llm_fallbacks():
    assert FakeLLM("Sure, here is a poem about the sea:\n\nWaves...").format("write me a poem about the sea") == \
        "Write me a poem about the sea."
    assert FakeLLM(RuntimeError("boom")).format("hola mundo esto es una prueba larga de texto") == \
        "Hola mundo esto es una prueba larga de texto."
    assert FakeLLM("Hola, mundo.").format("hola mundo") == "Hola, mundo."


def test_restore_breaks():
    pre = "update on the app.\nsecond thing the login fix is done\n\ny por último revisar el copy"
    out = "Update on the app. Second thing, the login fix is done.\nY por último, revisar el copy."
    assert _restore_breaks(pre, out) == "Update on the app.\nSecond thing, the login fix is done.\n\nY por último, revisar el copy."
    # already correct → unchanged
    good = "Update on the app.\nSecond thing, the login fix is done.\n\nY por último, revisar el copy."
    assert _restore_breaks(pre, good) == good
    assert _restore_breaks("no breaks here", "No breaks here.") == "No breaks here."


def test_needs_llm_heuristic():
    from dictdact.format import needs_llm
    assert not needs_llm("Which microphone is selected?")
    assert not needs_llm("Y es bastante tardado, no sé si es por el modelo que tenemos.")
    assert needs_llm("um so the the plan is to ship on monday")
    assert needs_llm("oye este puedes mandarme el reporte")
    assert needs_llm("primero el presupuesto segundo el roadmap")
    assert needs_llm("hola nueva línea mundo")
    assert needs_llm(" ".join(["palabra"] * 30))


# -- verbatim enforcement & result styles (the 09-03 incident class) ----------------
from dictdact.format import (STYLES, SYSTEM_PROMPT, VERBATIM_MIN_KEPT, _LLMFormatter,
                             language_flipped, style_prompt, words_kept)

RAW_LONG_ES = ("quiero que me ayudes a describir mi empresa la agencia hace automatizaciones "
               "agentes locales y consultoría para que los negocios adopten inteligencia artificial "
               "vendemos proyectos privados y planes mensuales con soporte")


def test_words_kept_faithful_cleanup_scores_high():
    assert words_kept(RAW_LONG_ES, RAW_LONG_ES.capitalize() + ".") > 0.9


def test_words_kept_summary_scores_low():
    assert words_kept(RAW_LONG_ES, "La empresa ofrece IA y soporte.") < VERBATIM_MIN_KEPT


def test_words_kept_translation_scores_low():
    out = ("I want you to help me describe my company the agency does automations local agents "
           "and consulting so businesses adopt artificial intelligence")
    assert words_kept(RAW_LONG_ES, out) < VERBATIM_MIN_KEPT


def test_words_kept_short_takes_have_no_signal():
    assert words_kept("hola qué tal", "hi how are you") == 1.0


def test_language_flip_detected_on_a_short_translation():
    assert language_flipped("¿que cuando quiero esto para más gracias?",
                            "what do they want when very thanks?")


def test_language_stable_mixed_take_not_flagged():
    pre = "quiero que el build esté listo para el viernes and then we ship gracias"
    out = "Quiero que el build esté listo para el viernes, and then we ship. Gracias."
    assert not language_flipped(pre, out)


class _CannedLLM(_LLMFormatter):
    name = "canned"
    always = True

    def __init__(self, reply):
        self._reply = reply

    def _complete(self, raw):
        return self._reply


# Long enough to sail past the legacy 0.35 length floor (the real incident scored 0.42
# there) — only the retention guard can catch it.
MARKETING_REWRITE = ("Somos líderes en transformación digital, impulsando el futuro del trabajo "
                     "con tecnología de punta y resultados medibles para cada cliente moderno.")


def test_clean_style_rejects_a_rewrite_and_returns_the_speakers_words():
    from dictdact.format import apply_spoken_commands, output_is_sane

    pre = apply_spoken_commands(RAW_LONG_ES)
    assert output_is_sane(pre, MARKETING_REWRITE)          # the OLD guards let it through…
    assert words_kept(pre, MARKETING_REWRITE) < VERBATIM_MIN_KEPT   # …retention does not
    f = _CannedLLM(MARKETING_REWRITE)
    out = f.format(RAW_LONG_ES)
    assert "empresa" in out                 # deterministic fallback kept the speaker's words
    assert "transformación digital" not in out
    assert f.last_fallback == "the model rewrote the take"


RAW_COMMANDS = ("o sea necesito que el equipo revise el presupuesto coma el roadmap coma digo "
                "perdón mejor dicho el plan de marketing punto primero contratar al diseñador "
                "segundo lanzar la campaña de navidad con veinte por ciento de descuento punto "
                "mándalo a laura arroba empresa punto com antes del reporte financiero")


def test_faithful_cleanup_with_spoken_commands_is_accepted():
    # Rules 4-8 ORDER these transformations; they must not count as lost words.
    reply = ("Necesito que el equipo revise el presupuesto, el roadmap, el plan de marketing.\n"
             "1. Contratar al diseñador.\n2. Lanzar la campaña de Navidad con 20% de descuento.\n"
             "Mándalo a laura@empresa.com antes del reporte financiero.")
    assert words_kept(RAW_COMMANDS, reply) > 0.9
    f = _CannedLLM(reply)
    assert f.format(RAW_COMMANDS) == reply
    assert f.last_fallback is None


def test_words_kept_ignores_repetition_padding():
    rambling = RAW_LONG_ES + " " + RAW_LONG_ES + " " + RAW_LONG_ES
    assert words_kept(rambling, MARKETING_REWRITE) < VERBATIM_MIN_KEPT


def test_words_kept_is_accent_insensitive():
    pre = ("necesito la consultoria practica del articulo tecnico sobre automatizacion "
           "rapida evaluacion economica migracion completa integracion segura documentacion clara")
    out = ("Necesito la consultoría práctica del artículo técnico sobre automatización, "
           "rápida evaluación económica, migración completa, integración segura, documentación clara.")
    assert words_kept(pre, out) == 1.0


def test_style_preamble_is_stripped_not_discarded():
    f = _CannedLLM("Aquí tienes el resumen: - Punto uno importante.")
    f.style = "summary"
    assert f.format(RAW_LONG_ES) == "- Punto uno importante."
    g = _CannedLLM("Sure! The team needs the report by Friday for the quarterly review meeting.")
    g.style = "translate_en"
    assert g.format(RAW_LONG_ES).startswith("The team needs")


def test_language_flip_caught_despite_punctuation_and_low_signal():
    assert language_flipped("quiero el reporte, gracias.", "I want the report, thanks.")
    assert language_flipped("hazme un resumen de la reunión de ayer",
                            "Make me a summary of yesterday's meeting")


def test_transform_styles_get_a_bigger_token_budget():
    from dictdact.format import _token_budget

    raw = "x" * 400
    assert _token_budget(raw, "translate_es") > _token_budget(raw) * 1.4


def test_translate_style_accepts_the_same_transformation():
    f = _CannedLLM("I want you to help me describe my company; the agency does automations and consulting.")
    f.style = "translate_en"
    assert "describe my company" in f.format(RAW_LONG_ES)


def test_summary_style_accepts_a_short_output():
    f = _CannedLLM("- IA y soporte para negocios.")
    f.style = "summary"
    assert f.format(RAW_LONG_ES).startswith("- IA")


def test_style_prompts_and_knob():
    from dictdact.config import Config
    from dictdact.server import EDITABLE

    assert style_prompt("clean") == SYSTEM_PROMPT
    assert style_prompt("desconocido") == SYSTEM_PROMPT
    assert "English" in style_prompt("translate_en")
    assert Config().formatter_style == "clean"
    assert EDITABLE["formatter_style"] == list(STYLES)
