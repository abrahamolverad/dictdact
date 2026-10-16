"""Unit tests for dictdact.transcribe — prompt building, repeat collapsing, output cleaning.

Deliberately never constructs WhisperEngine/ParakeetEngine (their __init__ imports and loads
real ML models). Every function under test here is a pure module-level function.
"""
from dictdact.transcribe import SUPPORTED, _clean, build_prompt, collapse_repeats


def test_supported_languages():
    assert SUPPORTED == ("es", "en")


# -- build_prompt -------------------------------------------------------------------------

def test_build_prompt_locked_es_has_no_english_clause():
    p = build_prompt("es", locked=True)
    assert p.startswith("Hola")
    assert "and then" not in p


def test_build_prompt_locked_en_has_no_spanish_clause():
    p = build_prompt("en", locked=True)
    assert "y después" not in p
    assert "¿" not in p


def test_build_prompt_unlocked_allows_mixing():
    p = build_prompt("en", locked=False)
    assert "y después" in p


def test_build_prompt_unknown_or_none_language_falls_back_to_english():
    assert build_prompt("fr", True) == build_prompt("en", True)
    assert build_prompt(None, True) == build_prompt("en", True)
    assert build_prompt("xx", False) == build_prompt("en", False)


def test_build_prompt_ends_with_the_vocabulary_tail():
    p = build_prompt("es", True, terms=["Acme Labs", "ACML"])
    assert p.endswith("Acme Labs, ACML.")


# -- collapse_repeats ---------------------------------------------------------------------

def test_collapse_short_phrase_loop_keeps_three_copies_of_a_word():
    # a single word keeps its emphasis ("no, no, no"), the loop is gone
    assert collapse_repeats("No, no, no, no, no, no, no.") == "No, no, no."
    assert collapse_repeats("no " * 30 + "voy") == "no no no voy"


def test_collapse_multiword_phrase_loop():
    assert collapse_repeats(
        "I think also, I think, I think, I think, I think, I think, I think that we go."
    ) == "I think also, I think that we go."
    assert collapse_repeats("en la imagen " * 9 + "en un club") == "en la imagen en un club"


def test_collapse_repeated_sentence():
    assert collapse_repeats("Y luego, ¿cómo estás? " * 12 + "Bien.") == "Y luego, ¿cómo estás? Bien."
    assert collapse_repeats("Y luego, ¿cómo estás? " * 4 + "Bien.") == "Y luego, ¿cómo estás? Bien."


def test_emphasis_up_to_five_repeats_is_preserved():
    # real emphatic speech must survive the guard
    s = "No, no, no, no, no voy a aceptar eso."
    assert collapse_repeats(s) == s
    assert collapse_repeats("no no no (only three) no") == "no no no (only three) no"
    assert collapse_repeats("I think, I think, I think, I think, I think that we go.") == \
        "I think, I think, I think, I think, I think that we go."


def test_up_to_three_identical_sentences_are_kept():
    assert collapse_repeats("No. No. Sí.") == "No. No. Sí."
    assert collapse_repeats("No. No. No. Sí.") == "No. No. No. Sí."


def test_collapse_repeats_empty_string():
    assert collapse_repeats("") == ""


# -- _clean ---------------------------------------------------------------------------------

def test_clean_strips_echoed_prompt_and_vocabulary_tail():
    terms = ["Acme Labs", "ACML", "Hyrox"]
    prompt = build_prompt("es", True, terms)
    tail = ", ".join(terms) + "."
    assert _clean("Acme Labs, ACML, Hyrox. hola mundo", prompt, tail) == "hola mundo"
    # a full prompt echo (two or more of its sentences) is stripped
    assert _clean("Hola, ¿cómo va todo? Vamos a revisar el plan para mañana y después mandamos el update al equipo.",
                  prompt, tail) == ""


def test_clean_keeps_a_real_dictation_that_matches_one_prompt_sentence():
    # "Hola, ¿cómo va todo?" is something people say
    es, en = build_prompt("es", True), build_prompt("en", True)
    assert _clean("Hola, ¿cómo va todo?", es) == "Hola, ¿cómo va todo?"
    assert _clean("Hi, how is everything?", en) == "Hi, how is everything?"
    assert _clean("Hola, ¿cómo va todo? Te mando el reporte hoy.", es) == "Hola, ¿cómo va todo? Te mando el reporte hoy."


def test_clean_drops_hallucinations():
    assert _clean("Thank you.") == ""


def test_clean_collapses_repeats_too():
    assert _clean("no, no, no, no, no, no, no, no, no, no") == "no, no, no"


def test_context_terms_keep_their_reserved_slots_in_the_prompt():
    from types import SimpleNamespace
    from dictdact.transcribe import WhisperEngine
    eng = SimpleNamespace(terms=[f"Term{i}" for i in range(45)], DICT_SLOTS=WhisperEngine.DICT_SLOTS,
                          CONTEXT_SLOTS=WhisperEngine.CONTEXT_SLOTS)
    terms = WhisperEngine._all_terms(eng, ["Simone", "Hyrox", "Term3"])
    # 3 context terms → the dictionary gets the 5 unused context slots (32 + 5)
    assert len([t for t in terms if t.startswith("Term")]) == WhisperEngine.DICT_SLOTS + 5
    assert "Simone" in terms and "Hyrox" in terms and terms.count("Term3") == 1
    assert len(terms) <= WhisperEngine.DICT_SLOTS + WhisperEngine.CONTEXT_SLOTS
    # 8+ context terms → exactly 32 dictionary terms and 8 context terms
    many = WhisperEngine._all_terms(eng, [f"Ctx{i}" for i in range(12)])
    assert len([t for t in many if t.startswith("Term")]) == WhisperEngine.DICT_SLOTS
    assert len([t for t in many if t.startswith("Ctx")]) == WhisperEngine.CONTEXT_SLOTS
    # no context → the dictionary keeps all 40 slots
    assert len(WhisperEngine._all_terms(eng, [])) == WhisperEngine.DICT_SLOTS + WhisperEngine.CONTEXT_SLOTS
